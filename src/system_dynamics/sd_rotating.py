import math

import numpy as np
import pandas as pd
from src.astrodynamic_functions import kepler_dynamics as kds
from src.astrodynamic_functions.kepler_dynamics import GRAV_CONST

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

    def load_model(self):

        self.names = ["body_1", "body_2"]
        self.masses = [None, None]
        self.SMAs = [None, None]
        self.ECCs = [None, None]
        self.INCs = [None, None]
        self.RAANs = [None, None]
        self.APERIs = [None, None]
        self.TAEPOs = [None, None]

    def propagate_body_states(self, times: list, mass, body_list=None, position_only=False):

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

        # body_states = self.propagate_body_states([system_time], None)
        x_acc, y_acc, z_acc = kds.CR3BP_acceleration(x, y, z, vx, vy, self.mass_parameter)

        acc_vector = [x_acc, y_acc, z_acc]

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

                acc_vector = [acc_vector[i] + command[i] for i in range(3)]

        return acc_vector

    def get_plotting_track(self, mass=None):

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