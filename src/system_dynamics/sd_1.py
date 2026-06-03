import copy
import math

import numpy as np
import pandas as pd
from src.astrodynamic_functions import kepler_dynamics as kds
from src.astrodynamic_functions.kepler_dynamics import GRAV_CONST

"""A force model that accepts a set of orbital parameters """


class inertial_force_model:
    def __init__(self, path):

        # Load the dataset that defines the bodies, WITHOUT the central attractor
        self.path_to_data = path
        self.names = []
        self.masses = []
        self.SMAs = []
        self.ECCs = []
        self.INCs = []
        self.RAANs = []
        self.APERIs = []
        self.TAEPOs = []
        self.trajectory_track: dict = {}
        self.plot_trajectory_track: dict = {}
        self.get_dataset()
        self.central_attractor_gravity_law = kds.gravitational_law

        # Central attractor
        self.central_mass = 0
        self.central_attractor_pos = [0, 0, 0]

        self.is_CR3BP = False

        # Steering
        self.guidance = None
        self.steer_acc_x = []
        self.steer_acc_y = []
        self.steer_acc_z = []
        self.true_time = []

        # Solar radiation pressure
        self.solar_pressure = None
        self.radiation_location = None
        self.tilt_angle = []
        self.clock_angle = []

        self.control_input = dict()

    def get_dataset(self):
        orbital_dataset = pd.read_excel(self.path_to_data)

        self.names = orbital_dataset["Name"]
        self.masses = [float(i) for i in orbital_dataset["Mass"]]
        self.SMAs = [float(i) for i in orbital_dataset["SMA"]]
        self.ECCs = [float(i) for i in orbital_dataset["ECC"]]
        self.INCs = [float(i) for i in orbital_dataset["INC"]]
        self.RAANs = [float(i) for i in orbital_dataset["RAAN"]]
        self.APERIs = [float(i) for i in orbital_dataset["APERI"]]
        self.TAEPOs = [float(i) for i in orbital_dataset["TAEPO"]]

    def define_central_attractor(self, mass, position: list):
        self.central_mass = mass
        self.central_attractor_pos = position

    def propagate_body_states(self, times: list, mass, body_list=None, position_only=False):
        if not body_list:
            body_list = self.names
        states = dict()
        if position_only:
            for k in range(len(body_list)):
                x_lst = []
                y_lst = []
                z_lst = []

                for time in times:
                    s_vec = kds.oe_to_sv(self.SMAs[k], self.ECCs[k], self.INCs[k], self.RAANs[k], self.APERIs[k],
                                           self.TAEPOs[k], time, mass)
                    x_lst.append(s_vec[0])
                    y_lst.append(s_vec[1])
                    z_lst.append(s_vec[2])

                body_dataset = [x_lst, y_lst, z_lst]
                states[body_list[k]] = body_dataset

                # Safe the trajectories to the class object as well
                self.trajectory_track = states
            return states
        else:
            for k in range(len(body_list)):
                body_dataset = [[], [], [], [], [], []]

                for time in times:
                    state = kds.oe_to_sv(self.SMAs[k], self.ECCs[k], self.INCs[k], self.RAANs[k], self.APERIs[k],
                                         self.TAEPOs[k], time, mass)
                    for i in range(6):
                        body_dataset[i].append(state[i])

                states[body_list[k]] = body_dataset

                # Safe the trajectories to the class object as well !!!NOT IN THE CASE OF VELOCITY DETERMINATION AS WELL
                # self.trajectory_track = states
            return states

    def get_acceleration(self, position, velocity, system_time):
        # self.singularities = []

        x = position[0]
        y = position[1]
        z = position[2]

        acc_vector = [0, 0, 0]

        body_states = self.propagate_body_states([system_time], mass=self.central_mass, position_only=True)
        body_names = list(body_states.keys())

        for k in range(len(body_names)):
            body_x = body_states[body_names[k]][0][0]
            body_y = body_states[body_names[k]][1][0]
            body_z = body_states[body_names[k]][2][0]

            x_acc, y_acc, z_acc = kds.gravitational_law(self.masses[k], x - body_x, y - body_y, z - body_z)
            acc_vector = [acc_vector[0] + x_acc, acc_vector[1] + y_acc, acc_vector[2] + z_acc]

        # Add acceleration from central attractor
        x_acc, y_acc, z_acc = self.central_attractor_gravity_law(self.central_mass,
                                                                 x - self.central_attractor_pos[0],
                                                                 y - self.central_attractor_pos[1],
                                                                 z - self.central_attractor_pos[2]
                                                                 )

        acc_vector = [acc_vector[0] + x_acc, acc_vector[1] + y_acc, acc_vector[2] + z_acc]

        """if self.solar_pressure:
            # The acceleration as a result of SRP is also written to the steering acceleration that is used as well for thruster control
            srp_acc, _, _ = self.solar_pressure.solar_acceleration(state=position)
            acc_vector = [acc_vector[0] + srp_acc[0], acc_vector[1] + srp_acc[1], acc_vector[2] + srp_acc[2]]

            self.steer_acc_x.append(srp_acc[0])
            self.steer_acc_y.append(srp_acc[1])
            self.steer_acc_z.append(srp_acc[2])

            self.tilt_angle.append(self.solar_pressure.sail_control[0])
            self.clock_angle.append(self.solar_pressure.sail_control[1])

            self.true_time.append(system_time)"""

        if self.guidance:
            #  Adding the acceleration vector from the steering law
            command = self.guidance.guidance(state=[x, y, z, velocity[0], velocity[1], velocity[2]], time=system_time,
                                             force_model=self)
            acc_vector = [acc_vector[0] + command[0], acc_vector[1] + command[1], acc_vector[2] + command[2]]

            self.steer_acc_x.append(command[0])
            self.steer_acc_y.append(command[1])
            self.steer_acc_z.append(command[2])

            self.true_time.append(system_time)

        # return [acc_vector[0], acc_vector[1], acc_vector[2]]
        return acc_vector

    def get_plotting_track(self, mass):
        MY = GRAV_CONST * mass
        states = dict()
        for k in range(len(self.names)):
            x_lst = []
            y_lst = []
            z_lst = []

            # Get the times based on the true anomaly
            SMA = self.SMAs[k]
            # terminal_time = 2 * math.pi * ((SMA ** 3) / (GRAV_CONST * self.central_mass)) ** 0.5
            # terminal_time = 2 * math.pi * ((SMA ** 3) / MY) ** 0.5
            terminal_time = 2 * math.pi * ((SMA ** 1.5) / (MY ** 0.5))
            times = list(np.linspace(0, terminal_time, 200))

            for time in times:
                s_vec = kds.oe_to_sv(self.SMAs[k], self.ECCs[k], self.INCs[k], self.RAANs[k], self.APERIs[k],
                                       self.TAEPOs[k], time, self.central_mass)
                x_lst.append(s_vec[0])
                y_lst.append(s_vec[1])
                z_lst.append(s_vec[2])

            body_dataset = [x_lst, y_lst, z_lst]
            states[self.names[k]] = body_dataset

            # Safe the trajectories to the class object as well
            self.plot_trajectory_track = states
        return states


class CR3BP:
    def __init__(self, mass_parameter):

        # Load the dataset that defines the bodies, WITHOUT the central attractor
        # self.path_to_data = path
        self.mass_parameter = mass_parameter
        self.names = []
        self.masses = []
        self.SMAs = []
        self.ECCs = []
        self.INCs = []
        self.RAANs = []
        self.APERIs = []
        self.TAEPOs = []
        self.trajectory_track: dict = {}
        self.plot_trajectory_track: dict = {}
        self.load_model()
        # self.central_attractor_gravity_law = kds.gravitational_law

        # Central attractor
        self.central_mass = 1000000000000
        self.central_attractor_pos = [0, 0, 0]

        # Lagrange point locations
        self.lagrange_points = dict()

        self.is_CR3BP = True

        # Steering
        self.steering_law = None
        self.steer_acc_x = []
        self.steer_acc_y = []
        self.steer_acc_z = []
        self.true_time = []

    def load_model(self):

        self.names = ["body_1", "body_2"]
        self.masses = [None, None]
        self.SMAs = [None, None]
        self.ECCs = [None, None]
        self.INCs = [None, None]
        self.RAANs = [None, None]
        self.APERIs = [None, None]
        self.TAEPOs = [None, None]

    def propagate_body_states(self, times: list):

        states = dict()
        for k in range(len(self.names)):
            x_lst = []
            y_lst = []
            z_lst = []

            for _ in times:
                if self.names[k] == "body_1":
                    x_lst.append(self.mass_parameter)
                else:
                    x_lst.append(1 - self.mass_parameter)
                y_lst.append(0.0)
                z_lst.append(0.0)

            body_dataset = [x_lst, y_lst, z_lst]
            states[self.names[k]] = body_dataset

            # Safe the trajectories to the class object as well
            self.trajectory_track = states
        return states

    def get_acceleration(self, position, velocity, system_time):
        # self.singularities = []

        x = position[0]
        y = position[1]
        z = position[2]

        vx = velocity[0]
        vy = velocity[1]

        body_states = self.propagate_body_states([system_time])
        x_acc, y_acc, z_acc = kds.CR3BP_acceleration(x, y, z, vx, vy, self.mass_parameter)

        acc_vector = [x_acc, y_acc, z_acc]

        if self.steering_law:
            #  Adding the acceleration vector from the steering law
            command = self.steering_law.guidance(state=[x, y, z, velocity[0], velocity[1], velocity[2]],
                                                 time=system_time, force_model=self)
            acc_vector = [acc_vector[0] + command[0], acc_vector[1] + command[1], acc_vector[2] + command[2]]

            self.steer_acc_x.append(command[0])
            self.steer_acc_y.append(command[1])
            self.steer_acc_z.append(command[2])

            self.true_time.append(system_time)

        return [acc_vector[0], acc_vector[1], acc_vector[2]]

    def get_plotting_track(self):

        states = dict()
        for k in range(len(self.names)):
            x_lst = []
            y_lst = []
            z_lst = []

            times = np.linspace(0, 1, 2).tolist()

            for _ in times:
                if self.names[k] == "body_1":
                    x_lst.append(self.mass_parameter)
                else:
                    x_lst.append(1 - self.mass_parameter)
                y_lst.append(0.0)
                z_lst.append(0.0)

            body_dataset = [x_lst, y_lst, z_lst]
            states[self.names[k]] = body_dataset

            # Safe the trajectories to the class object as well
            self.plot_trajectory_track = states
        return states

    def get_lagrange_points(self):
        self.lagrange_points["L1"] = [
            1 - ((self.mass_parameter ** 0.33071) / (0.51233 * self.mass_parameter ** 0.49128 + 1.487864)), 0, 0]
        self.lagrange_points["L2"] = [1 + ((self.mass_parameter ** 0.8383 + 2.891 * self.mass_parameter ** 0.3358) / (
                    1.525 * self.mass_parameter ** 0.848 + 4.046596)), 0, 0]
        self.lagrange_points["L3"] = [
            -1 + ((self.mass_parameter ** 1.007) / (1.653 * self.mass_parameter ** 0.9375 + 1.66308)), 0, 0]
        self.lagrange_points["L4"] = [0.5 - self.mass_parameter, 0.5 * 3 ** 0.5, 0]
        self.lagrange_points["L5"] = [0.5 - self.mass_parameter, -0.5 * 3 ** 0.5, 0]
        pass

    def get_potential_field(self, resolution, lim_x, lim_y):
        x_arr = list(np.linspace(lim_x[0], lim_x[1], resolution))
        y_arr = list(np.linspace(lim_y[0], lim_y[1], resolution))
        xv, yv = np.meshgrid(np.linspace(lim_x[0], lim_x[1], resolution), np.linspace(lim_y[0], lim_y[1], resolution),
                             indexing='ij')

        potential_map, _ = np.meshgrid(np.linspace(lim_x[0], lim_x[1], resolution),
                                       np.linspace(lim_y[0], lim_y[1], resolution), indexing='ij')
        for i in range(resolution):
            for j in range(resolution):
                potential_map[i][j] = self.pseudo_potential(x_arr[i], y_arr[j])
        return list(xv), list(yv), list(potential_map)

    def pseudo_potential(self, x, y):
        r_1 = ((x + self.mass_parameter) ** 2 + y ** 2) ** 0.5
        r_2 = ((x - 1 + self.mass_parameter) ** 2 + y ** 2) ** 0.5

        V = - ((1 - self.mass_parameter) / r_1) - self.mass_parameter / r_2 - 0.5 * self.mass_parameter * (
                    1 - self.mass_parameter) - 0.5 * (x ** 2 + y ** 2)
        return V
