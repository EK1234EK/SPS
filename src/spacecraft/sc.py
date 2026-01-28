import time
import numpy as np
from numpy import interp
from scipy.integrate import solve_ivp
from src.astrodynamic_functions import kepler_dynamics
from src.astrodynamic_functions.kepler_dynamics import GRAV_CONST


class Spacecraft:
    def __init__(self, init_state_vector, force_model):
        self.init_state_vector = init_state_vector  # [x1_1, x2_1, x3_1, dx1, dx2, dx3]
        self.trajectory_track = [[], [], [], [], [], []]
        self.orbital_parameters_track = [[], [], [], [], [], []]
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

        self.plot_color = "map" # [1, 0, 1]
        self.tail_color = [0.5, 0.5, 0.5]
        self.display_name = ''

        self.is_feasible = False
        self.sensitivity = False

        self.condition_value = [0]

        # Steering track
        self.steer_x = []
        self.steer_y = []
        self.steer_z = []

    def get_acc(self, state_vector, system_time):
        position = state_vector[0:3]
        velocity = state_vector[3:6]
        self.acc_vec = self.force_model.get_acceleration(position, velocity, system_time)

        # print(self.acc_vec)
        velocity = list(velocity)
        velocity.extend(self.acc_vec)
        return velocity

    def trajectory_to_orbital_parameters(self, mass):
        # t_1 = time.time()

        for k in range(len(self.trajectory_track[0])):
            state_vector = [self.trajectory_track[0][k], self.trajectory_track[1][k], self.trajectory_track[2][k],
                            self.trajectory_track[3][k], self.trajectory_track[4][k], self.trajectory_track[5][k]]
            orbital_parameters = kepler_dynamics.sv_to_oe(state_vector, mass, -self.integration_points[k])
            # orbital_parameters = kepler_dynamics.sv_to_oe(state_vector, mass)

            for i in range(6):
                self.orbital_parameters_track[i].append(orbital_parameters[i])
        # t_2 = time.time()
        # print(self.display_name + ": Cartesian to Keplerian " + str(round(t_2 - t_1, 3)) + " s")

    def get_body_distances(self, body_list: list, body_trajectories = None):
        distances = dict()
        if body_trajectories is None:
            self.force_model.trajectory_track = self.force_model.propagate_body_states(self.integration_points)
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

    def integrate_states_sivp(self, method='DOP853', rtol=1e-3, atol=1e-6):
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

        def get_acc_wrapper(t, state_vector):
            return self.get_acc(state_vector, t)

        if self.time_interval[0] == self.integration_points[0] and self.time_interval[1] == self.integration_points[-1]:
            eval_points = self.integration_points
        else:
            eval_points = np.linspace(self.time_interval[0], self.time_interval[1],
                                      len(self.integration_points)).tolist()
        init_time = time.time()
        sol = solve_ivp(get_acc_wrapper,
                        t_span=[self.time_interval[0], self.time_interval[1]],
                        y0=self.init_state_vector,
                        method=method,
                        t_eval=eval_points,
                        atol=atol,
                        rtol=rtol
                        )
        terminal_time = time.time()
        steps = sol.nfev
        sol = list(sol.y)


        if self.time_interval[0] == self.integration_points[0] and self.time_interval[1] == self.integration_points[-1]:
            for state in range(6):
                self.trajectory_track[state] += sol[state].tolist()
        else:
            for state in range(6):
                sol[state] = interp(self.integration_points, eval_points, sol[state], left=None, right=None).tolist()
                self.trajectory_track[state] += sol[state]

        # Save the steering acceleration
        """self.steer_x = interp(self.integration_points, eval_points, self.force_model.steer_acc_x, left=None, right=None).tolist()
        self.steer_y = interp(self.integration_points, eval_points, self.force_model.steer_acc_y, left=None, right=None).tolist()
        self.steer_z = interp(self.integration_points, eval_points, self.force_model.steer_acc_z, left=None, right=None).tolist()"""

        self.steer_x = self.force_model.steer_acc_x
        self.steer_y = self.force_model.steer_acc_y
        self.steer_z = self.force_model.steer_acc_z

        print(self.display_name + ": Integrating " + str(round(terminal_time - init_time, 3)) + " s after " + str(
            steps) + " evaluations")

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
            self.trajectory_to_orbital_parameters(mass=mass)
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
