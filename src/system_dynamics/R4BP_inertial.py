import copy
import math

import numpy as np
import pandas as pd
from src.astrodynamic_functions import kepler_dynamics as kds
from src.astrodynamic_functions.kepler_dynamics import GRAV_CONST

"""A force model that accepts a set of orbital parameters """


class R4BP_force_model:
    def __init__(self, path):

        self.is_CR3BP = False

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

        # Atmospheric drag
        self.drag_model = None
        self.drag_acc_x = []
        self.drag_acc_y = []
        self.drag_acc_z = []
        self.drag_mag_track = []

        self.control_input = dict()

        # Pseudo graviataional parameters so that the orbital period is correct
        self.body_propagation_masses = {"Earth": 723833235.9367174 / (6.67430*10**(-11)), "Moon": 388972827096717.2 / (6.67430*10**(-11)), "Sun": 1.3271284354451499e+20 / (6.67430 * 10 ** (-11))}

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
                if body_list[k] not in self.body_propagation_masses.keys():
                    raise ValueError("Invalid body " + body_list[k])

                mass = self.body_propagation_masses[body_list[k]]

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
            # for k in range(len(body_list)):
            for body in body_list:
                k = list(self.names).index(body)
                if body not in self.body_propagation_masses.keys():
                    raise ValueError("Invalid body " + body)

                mass = self.body_propagation_masses[body]

                body_dataset = [[], [], [], [], [], []]

                for time in times:
                    state = kds.oe_to_sv(self.SMAs[k], self.ECCs[k], self.INCs[k], self.RAANs[k], self.APERIs[k],
                                         self.TAEPOs[k], time, mass)
                    for i in range(6):
                        body_dataset[i].append(state[i])

                states[body] = body_dataset

                # Safe the trajectories to the class object as well !!!NOT IN THE CASE OF VELOCITY DETERMINATION AS WELL
                # self.trajectory_track = states
            return states

    def get_acceleration(self, position, velocity, system_time):
        # self.singularities = []

        x = position[0]
        y = position[1]
        z = position[2]

        acc_vector = [0, 0, 0]

        body_states = self.propagate_body_states([system_time], mass=None, position_only=True)
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

        acc_vector = [acc_vector[0] + 0*x_acc, acc_vector[1] + 0*y_acc, acc_vector[2] + 0*z_acc]

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

            if self.drag_model:
                command = self.drag_model.get_aero_acc(state=np.concatenate((position, velocity)), n=self.guidance.current_n, sigma=self.solar_pressure.sail_parameters["sigma"])
                self.drag_acc_x.append(command[0])
                self.drag_acc_y.append(command[1])
                self.drag_acc_z.append(command[2])
                self.drag_mag_track.append(np.linalg.norm(command))

                acc_vector = [acc_vector[i] + command[i]for i in range(3)]

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
