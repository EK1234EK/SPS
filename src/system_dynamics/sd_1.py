import math

import numpy as np
import pandas as pd
# from jaraco.functools import retry

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

    def get_dataset(self):
        orbital_dataset = pd.read_excel(self.path_to_data)

        self.names = orbital_dataset["Name"]
        self.masses = orbital_dataset["Mass"]
        self.SMAs = orbital_dataset["SMA"]
        self.ECCs = orbital_dataset["ECC"]
        self.INCs = orbital_dataset["INC"]
        self.RAANs = orbital_dataset["RAAN"]
        self.APERIs = orbital_dataset["APERI"]
        self.TAEPOs = orbital_dataset["TAEPO"]

    def define_central_attractor(self, mass, position: list):
        self.central_mass = mass
        self.central_attractor_pos = position

    def propagate_body_states(self, times: list):

        states = dict()
        for k in range(len(self.names)):
            x_lst = []
            y_lst = []
            z_lst = []

            for time in times:
                x, y, z = kds.oe_to_sv(self.SMAs[k], self.ECCs[k], self.INCs[k], self.RAANs[k], self.APERIs[k],
                                       self.TAEPOs[k], time)
                x_lst.append(float(x))
                y_lst.append(float(y))
                z_lst.append(float(z))

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

        acc_vector = [0, 0, 0]

        body_states = self.propagate_body_states([system_time])
        body_names = list(body_states.keys())

        for k in range(len(body_names)):
            body_x = body_states[body_names[k]][0][0]
            body_y = body_states[body_names[k]][1][0]
            body_z = body_states[body_names[k]][2][0]

            x_acc, y_acc, z_acc = kds.gravitational_law(self.masses[k], x - body_x, y - body_y, z - body_z)
            acc_vector = [acc_vector[0] + x_acc, acc_vector[1] + y_acc, acc_vector[2] + z_acc]

        # Add acceleration from central attractor
        """x_acc, y_acc, z_acc = kds.gravitational_law(self.central_mass,
                                                    x - self.central_attractor_pos[0],
                                                    y - self.central_attractor_pos[1],
                                                    z - self.central_attractor_pos[2]
                                                    )"""

        x_acc, y_acc, z_acc = self.central_attractor_gravity_law(self.central_mass,
                                                                 x - self.central_attractor_pos[0],
                                                                 y - self.central_attractor_pos[1],
                                                                 z - self.central_attractor_pos[2]
                                                                 )

        acc_vector = [acc_vector[0] + x_acc, acc_vector[1] + y_acc, acc_vector[2] + z_acc]
        return [acc_vector[0], acc_vector[1], acc_vector[2]]

    def get_plotting_track(self):

        states = dict()
        for k in range(len(self.names)):
            x_lst = []
            y_lst = []
            z_lst = []

            # Get the times based on the true anomaly
            terminal_time = 2 * math.pi * (self.SMAs[k] ** 3 / (GRAV_CONST * self.central_mass)) ** 0.5
            times = np.linspace(0, terminal_time, 200).tolist()

            for time in times:
                x, y, z = kds.oe_to_sv(self.SMAs[k], self.ECCs[k], self.INCs[k], self.RAANs[k], self.APERIs[k],
                                       self.TAEPOs[k], time)
                x_lst.append(float(x))
                y_lst.append(float(y))
                z_lst.append(float(z))

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
        return [x_acc, y_acc, z_acc]

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
        self.lagrange_points["L1"] = [1 - ((self.mass_parameter**0.33071) / (0.51233 * self.mass_parameter**0.49128 + 1.487864)), 0, 0]
        self.lagrange_points["L2"] = [1 + ((self.mass_parameter**0.8383 + 2.891 * self.mass_parameter**0.3358) / (1.525 * self.mass_parameter**0.848 + 4.046596)), 0, 0]
        self.lagrange_points["L3"] = [-1 + ((self.mass_parameter**1.007) / (1.653 * self.mass_parameter**0.9375 + 1.66308)), 0, 0]
        self.lagrange_points["L4"] = [0.5 - self.mass_parameter, 0.5 * 3**0.5, 0]
        self.lagrange_points["L5"] = [0.5 - self.mass_parameter, -0.5 * 3**0.5, 0]
        pass
