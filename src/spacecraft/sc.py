import time
import numpy as np
from numpy import interp
from scipy.integrate import solve_ivp

import src.globals.colors
from src.astrodynamic_functions import kepler_dynamics
from src.astrodynamic_functions.kepler_dynamics import GRAV_CONST


class Spacecraft:
    def __init__(self, init_state_vector, force_model):
        self.init_state_vector = init_state_vector  # [x1_1, x2_1, x3_1, dx1, dx2, dx3]
        self.trajectory_track = [[], [], [], [], [], []]
        self.orbital_parameters_track = [[], [], [], [], [], []]
        self.orbital_parameters_reference = ""
        self.force_model = force_model
        self.acc_vec = []

        self.integration_points = []
        self.time_interval = []

        self.body_distances = dict()

        self.C3_track = []
        self.slant_range_track = []
        self.vel_mag_track = []

        # self.integration_time = math.nan
        self.is_edge_spacecraft: bool = False
        self.is_center_spacecraft: bool = False
        self.is_resampled_spacecraft: bool = False

        self.plot_color = "map"  # [1, 0, 1]
        self.tail_color = [0.5, 0.5, 0.5]
        self.display_name = ''

        self.is_feasible = False
        self.sensitivity = False

        self.condition_value = [0]

        self.event_time = None

        # Steering track
        self.steer_x = []
        self.steer_y = []
        self.steer_z = []
        self.steer_magnitude = []
        self.true_time = []

        # Sail control
        self.tilt = []
        self.clock = []
        self.control_input_track = dict()
        self.vel_angle_track = dict()

    def get_acc(self, state_vector, system_time):
        position = state_vector[0:3]
        velocity = state_vector[3:6]
        self.acc_vec = self.force_model.get_acceleration(position, velocity, system_time)

        # print(self.acc_vec)
        velocity = list(velocity)
        velocity.extend(self.acc_vec)
        return velocity

    def trajectory_to_orbital_parameters(self, conversion_mass, reference=None):
        # t_1 = time.time()
        if reference:
            if reference not in self.force_model.names.to_list():
                raise ValueError("Name " + reference + " not in force model!")
            body_states = self.force_model.propagate_body_states(times=self.integration_points, mass=self.force_model.central_mass, body_list=[reference])[reference]
            for k in range(len(self.trajectory_track[0])):
                state_vector = [self.trajectory_track[i][k] - body_states[i][k] for i in range(6)]
                orbital_parameters = kepler_dynamics.sv_to_oe(state_vector, conversion_mass, -self.integration_points[k])
                for i in range(6):
                    self.orbital_parameters_track[i].append(orbital_parameters[i])
            self.orbital_parameters_reference = reference
        else:
            for k in range(len(self.trajectory_track[0])):
                state_vector = [self.trajectory_track[i][k] for i in range(6)]
                orbital_parameters = kepler_dynamics.sv_to_oe(state_vector, conversion_mass, -self.integration_points[k])

                for i in range(6):
                    self.orbital_parameters_track[i].append(orbital_parameters[i])
        # t_2 = time.time()
        # print(self.display_name + ": Cartesian to Keplerian " + str(round(t_2 - t_1, 3)) + " s")

    def get_body_distances(self, body_list: list, body_trajectories=None):
        distances = dict()
        if body_trajectories is None:
            self.force_model.trajectory_track = self.force_model.propagate_body_states(self.integration_points, self.force_model.central_mass)
        else:
            self.force_model.trajectory_track = body_trajectories

        for body in body_list:
            ind_dist = [0 for k in range(len(self.force_model.trajectory_track[body][0]))]
            for k in range(len(self.force_model.trajectory_track[body][0])):
                dx = self.force_model.trajectory_track[body][0][k] - self.trajectory_track[0][k]
                dy = self.force_model.trajectory_track[body][1][k] - self.trajectory_track[1][k]
                dz = self.force_model.trajectory_track[body][2][k] - self.trajectory_track[2][k]

                ind_dist[k] = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            distances[body] = ind_dist
        self.body_distances = distances

    def integrate_states_sivp(self, method='DOP853', rtol=1e-3, atol=1e-6, terminator=None):
        """
        :param method: RK45, RK23, DOP853, Radau, BDF, LSODA
        :param rtol:
        :param atol:
        :return:
        """

        # self.integration_points = time_vector

        # Reset the steering acceleration from the force model
        self.force_model.steer_acc_x = []
        self.force_model.steer_acc_y = []
        self.force_model.steer_acc_z = []
        self.force_model.true_time = []
        self.force_model.tilt_angle = []
        self.force_model.clock_angle = []

        if self.force_model.guidance:
            self.force_model.guidance.control_command_track = dict()
            self.force_model.guidance.vel_angle_track = dict()

        def get_acc_wrapper(t, state_vector):
            return self.get_acc(state_vector, t)

        print(self.display_name + ": ", end="")

        if self.time_interval[0] == self.integration_points[0] and self.time_interval[1] == self.integration_points[-1]:
            eval_points = self.integration_points
        else:
            eval_points = np.linspace(self.time_interval[0], self.time_interval[1],
                                      len(self.integration_points)).tolist()
        init_time = time.time()

        terminator.terminal = True

        sol = solve_ivp(get_acc_wrapper,
                        t_span=[self.time_interval[0], self.time_interval[1]],
                        y0=self.init_state_vector,
                        method=method,
                        t_eval=eval_points,
                        atol=atol,
                        rtol=rtol,
                        events=terminator,
                        dense_output=True
                        )
        terminal_time = time.time()
        steps = sol.nfev
        status = sol.success
        integrated_time = sol.t
        terminal_message = sol.message

        if terminal_message == 'A termination event occurred.':
            # print("Escape: ", sol.t_events[0][0] / (24*3600))
            self.time_interval = [integrated_time[0], integrated_time[-1]]
            self.event_time = sol.t_events

        sol = list(sol.y)

        if self.time_interval[0] == self.integration_points[0] and self.time_interval[1] == self.integration_points[
            -1] and status == True and terminal_message != 'A termination event occurred.':
            for state in range(6):
                self.trajectory_track[state] += sol[state].tolist()
        elif status is not True:
            self.display_name += " - Integration error"
            for state in range(6):
                sol[state] = interp(self.integration_points, integrated_time, sol[state], left=None,
                                    right=None).tolist()
                self.trajectory_track[state] += sol[state]
        elif terminal_message == 'A termination event occurred.':
            for state in range(6):
                sol[state] = interp(self.integration_points, integrated_time, sol[state], left=np.nan, right=np.nan).tolist()
                sol[state] = [None if x == np.nan else x for x in sol[state]]
                self.trajectory_track[state] += sol[state]
        else:
            for state in range(6):
                sol[state] = interp(self.integration_points, eval_points, sol[state], left=None, right=None).tolist()
                self.trajectory_track[state] += sol[state]

        # Save the steering acceleration
        if self.force_model.guidance is not None:
            print("Guiding, ", end="")

            self.steer_magnitude = [(self.force_model.steer_acc_x[i] ** 2 + self.force_model.steer_acc_y[i] ** 2 +
                                     self.force_model.steer_acc_z[i] ** 2) ** 0.5 for i in
                                    range(len(self.force_model.steer_acc_x))]

            self.steer_x = interp(self.integration_points, self.force_model.true_time, self.force_model.steer_acc_x,
                                  left=None, right=None).tolist()
            self.steer_y = interp(self.integration_points, self.force_model.true_time, self.force_model.steer_acc_y,
                                  left=None, right=None).tolist()
            self.steer_z = interp(self.integration_points, self.force_model.true_time, self.force_model.steer_acc_z,
                                  left=None, right=None).tolist()

            steer_mag_res = interp(self.integration_points, self.force_model.true_time, self.steer_magnitude,
                                   left=None, right=None).tolist()

            self.steer_magnitude = steer_mag_res

            """if self.force_model.tilt_angle:
                print(" - Sail control: ", end="")
    
                self.tilt = interp(self.integration_points, self.force_model.true_time, self.force_model.tilt_angle,
                                   left=None, right=None).tolist()
                self.clock = interp(self.integration_points, self.force_model.true_time, self.force_model.clock_angle,
                                    left=None, right=None).tolist()"""

            self.control_input_track = self.force_model.guidance.control_command_track
            self.vel_angle_track = self.force_model.guidance.vel_angle_track

            # Interpolating everything onto the integration points:
            for key in self.control_input_track.keys():
                self.control_input_track[key] = interp(self.integration_points, self.force_model.true_time,
                                                       self.control_input_track[key], left=None, right=None).tolist()

            for key in self.vel_angle_track.keys():
                self.vel_angle_track[key] = interp(self.integration_points, self.force_model.true_time,
                                                       self.vel_angle_track[key], left=None, right=None).tolist()

        print("Integrating " + str(round(terminal_time - init_time, 3)) + " s after " + str(
            steps) + " evaluations", end="")

        print()

    def get_C3_track(self, central_mass):
        if len(self.orbital_parameters_track[0]) == 0:
            self.trajectory_to_orbital_parameters(central_mass)

        for a in self.orbital_parameters_track[0]:
            if a is None:
                self.C3_track.append(None)
            else:
                self.C3_track.append(-central_mass * GRAV_CONST / a)

    def get_state_magnitudes(self):
        self.slant_range_track = []
        self.vel_mag_track = []

        for x, y, z, vx, vy, vz in zip(self.trajectory_track[0], self.trajectory_track[1],
                                       self.trajectory_track[2], self.trajectory_track[3],
                                       self.trajectory_track[4], self.trajectory_track[5]):
            self.slant_range_track.append((x ** 2 + y ** 2 + z ** 2) ** 0.5)
            self.vel_mag_track.append((vx ** 2 + vy ** 2 + vz ** 2) ** 0.5)

    def trajectory_conversion(self, mass, transformations=("Kepler", "C3", "State_magnitude")):
        if "Kepler" in transformations:
            self.trajectory_to_orbital_parameters(conversion_mass=mass)
        if "C3" in transformations:
            self.get_C3_track(central_mass=mass)
        if "State_magnitude" in transformations:
            self.get_state_magnitudes()
        else:
            raise ValueError(
                "Error: Invalid trajectory transfomration qualifier! Use 'Kepler', 'C3 or 'State_magnitude'")

    def revert_trajectory(self):
        for i in range(6):
            self.trajectory_track[i] = list(reversed(self.trajectory_track[i]))
