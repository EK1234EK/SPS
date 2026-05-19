import copy

import numpy as np
import math
from src.system_dynamics import sail_repository

from matplotlib.projections import projection_registry

import src.globals.Constants

sigma_star, L_s, R_s, c = src.globals.Constants.get_SRP_globals()
G, MY, KS_TOLERANCE, GRAV_CONST, EARTH_RADIUS = src.globals.Constants.get_globals()


def inverse_square_SRP(tilt_angle, radiation_location, sail_loading, central_attractor_mass, state):
    distance = np.linalg.norm(np.array(state[0:3]) - np.array(radiation_location))
    acc = (sigma_star / sail_loading) * (
            central_attractor_mass * GRAV_CONST / distance ** 2)  #  * math.cos(tilt_angle)**2
    return acc


def modified_inverse_square_SRP(radiation_location, sail_loading, state):
    distance = np.linalg.norm(np.array(state[0:3]) - np.array(radiation_location))
    press = (L_s / (3 * math.pi * c * R_s ** 2)) * (1 - (1 - (R_s / distance) ** 2) ** 1.5)
    acc = (press / sail_loading)  #  * math.cos(tilt_angle)**2
    return acc


def sail_attitude(sail_control, radiation_location, state):
    alpha = sail_control[0]  # Tilt angle: Alpha, acts around the second axis
    gamma = sail_control[1]  # Clock angle: Gamma, acts around the inbound radiation axis

    d_1 = np.array(state) - np.array(radiation_location)
    d_1 = d_1 / np.linalg.norm(d_1)

    d_2 = np.cross(d_1, np.array([0, 0, 1]))  # I'm sure this will lead to problems later on
    d_2 = d_2 / np.linalg.norm(d_2)

    d_3 = np.cross(d_2, d_1)
    d_3 = d_3 / np.linalg.norm(d_3)

    n = math.cos(alpha) * d_1 + math.sin(alpha) * math.sin(gamma) * d_2 + math.sin(alpha) * math.cos(gamma) * d_3
    return d_1, d_2, d_3, n


def quadrant_checking(alpha: float, sail_parameters: dict):
    while alpha > 0.5 * math.pi:
        alpha -= math.pi
        sail_parameters = flip(sail_parameters)
    while alpha < -0.5 * math.pi:
        alpha += math.pi
        sail_parameters = flip(sail_parameters)
    return alpha, sail_parameters["r_f"],  sail_parameters["r_b"], sail_parameters["s_f"], sail_parameters["s_b"], sail_parameters["B_f"], sail_parameters["B_b"], sail_parameters["e_f"], sail_parameters["e_b"]


def flip(sail_parameters):
    new_params = {
        "sigma": 1,
        "r_f": sail_parameters["r_b"],
        "r_b": sail_parameters["r_f"],
        "s_f": sail_parameters["s_b"],
        "s_b": sail_parameters["s_f"],
        "B_f": sail_parameters["B_b"],
        "B_b": sail_parameters["B_f"],
        "e_f": sail_parameters["e_b"],
        "e_b": sail_parameters["e_f"]
    }
    return new_params


class Solar_pressure:

    def __init__(self, sail_model, central_attractor_mass):
        self.sail_parameters = sail_repository.Sail_parameters().fetch(name=sail_model)
        self.radiation_location = [0, 0, 0]

        self.sail_control = [0, 0]
        self.sail_loading = self.sail_parameters["sigma"]
        self.central_attractor_mass = central_attractor_mass

    def ideal_sail(self, state):

        d_1, d_2, d_3, n = sail_attitude(sail_control=self.sail_control, radiation_location=self.radiation_location,
                                         state=state)
        # base_acc = modified_inverse_square_SRP(radiation_location=self.radiation_location, sail_loading=self.sail_loading, state=state)
        base_acc = inverse_square_SRP(tilt_angle=self.sail_control[0], radiation_location=self.radiation_location,
                                      sail_loading=self.sail_loading,
                                      central_attractor_mass=self.central_attractor_mass, state=state)

        # Taking into account the direction of the resulting acceleration:

        if np.dot(d_1, n) > 0:
            acc = list(base_acc * (np.dot(d_1, n)) ** 2 * n)
        else:
            acc = list(-base_acc * (np.dot(d_1, n)) ** 2 * n)
        return acc, 0, 0

    def real_sail(self, state):

        alpha = self.sail_control[0]
        clock = self.sail_control[1]

        # Parameter defintions:
        P = modified_inverse_square_SRP(radiation_location=self.radiation_location, sail_loading=self.sail_loading,
                                        state=state) * self.sail_loading

        d_1, d_2, d_3, n = sail_attitude(sail_control=[alpha, clock], radiation_location=self.radiation_location,
                                         state=state)

        # Making sure the acceleration only acts away from the sun
        alpha, r,  _, s, _, B_f, B_b, epsilon_f, epsilon_B = quadrant_checking(alpha, self.sail_parameters)

        # Determining the t-vector

        incidence = np.array(state[0:3]) - np.array(self.radiation_location)
        temp = np.cross(n, incidence)
        if np.linalg.norm(temp) == 0:
            t = n
        else:
            t = (d_1 - math.cos(alpha) * n) / math.sin(alpha)  # TODO Check if minus in numerator instead

        # Determining the two pre-factors:
        # Normal force
        f_n_mag = P * ((1 + r * s) * math.cos(alpha) ** 2 + B_f * (1 - s) * r * math.cos(alpha) + (1 - r) * math.cos(
            alpha) * (
                               epsilon_f * B_f - epsilon_B - B_b) / (epsilon_f + epsilon_B))
        f_n = f_n_mag * n

        # Transversal force
        f_t_mag = P * (1 - r * s) * math.cos(alpha) * math.sin(alpha)
        f_t = f_t_mag * t

        acc = (f_n + f_t) / self.sail_loading
        """print("n: ", n)
        print("t: ", t)
        print("Acc: ", acc)  # --> TODO this needs fixing it is negative, check clock angle first!
        print("f_n: ", f_n)
        print("f_n_mag: ", f_n_mag)
        print("f_t: ", f_t)
        print("f_t_mag: ", f_t_mag)
        print()
        print("====================")"""
        return acc, f_n, f_t

    def solar_acceleration(self, state):
        acc, _, _ = self.ideal_sail(state=state)
        return acc


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'


    def cartesian_surface_plots():

        SRP = Solar_pressure(sail_model="ACS3", central_attractor_mass=1.989 * 10 ** 30)
        state = [149*10**9, 0, 0]

        def get_acc(tilt, clock):
            SRP.sail_control = [tilt, clock]
            acc = SRP.solar_acceleration(state=state)
            return acc

        tilt_angle = np.linspace(0 * math.pi, 2 * math.pi, 50)
        clock_angle = np.linspace(0 * math.pi, 2 * math.pi, 50)

        # acc_array = list(np.zeros(shape=(len(tilt_angle), len(clock_angle))))

        acc_x, acc_y, acc_z, TILT, CLOCK, ANGLE = (np.zeros(shape=(len(tilt_angle), len(clock_angle))) for _ in
                                                   range(6))

        for i, tilt in enumerate(tilt_angle):
            for j, clock in enumerate(clock_angle):
                acc_vec = get_acc(tilt, clock)
                TILT[i][j] = tilt * 180 / math.pi
                CLOCK[i][j] = clock * 180 / math.pi

                acc_x[i][j] = acc_vec[0]
                acc_y[i][j] = acc_vec[1]
                acc_z[i][j] = acc_vec[2]

                ANGLE[i][j] = math.acos((np.dot(acc_vec, np.array(state) - np.array(SRP.radiation_location))) / (
                        np.linalg.norm(np.array(acc_vec)) * np.linalg.norm(
                    np.array(state) - np.array(SRP.radiation_location))))

        fig_1 = plt.figure()
        fig_2 = plt.figure()
        fig_3 = plt.figure()
        fig_4 = plt.figure()
        ax_1 = fig_1.add_subplot(111, projection="3d", )
        ax_2 = fig_2.add_subplot(111, projection="3d")
        ax_3 = fig_3.add_subplot(111, projection="3d")
        ax_4 = fig_4.add_subplot(111, projection="3d")

        plt_1 = ax_1.plot_surface(TILT, CLOCK, acc_x, label="Acceleration x", cmap=mpl.colormaps["hsv"])
        plt_2 = ax_2.plot_surface(TILT, CLOCK, acc_y, label="Acceleration y", cmap=mpl.colormaps["hsv"])
        plt_3 = ax_3.plot_surface(TILT, CLOCK, acc_z, label="Acceleration z", cmap=mpl.colormaps["hsv"])
        plt_4 = ax_4.plot_surface(TILT, CLOCK, ANGLE * 180 / math.pi, label="Acceleration angle",
                                  cmap=mpl.colormaps["hsv"])

        ax_1.set_title("X acceleration")
        ax_2.set_title("Y acceleration")
        ax_3.set_title("Z acceleration")
        ax_4.set_title("Acceleration angle")

        axes = [ax_1, ax_2, ax_3, ax_4]
        for axis in axes:
            axis.set_xlabel("Tilt angle [rad]")
            axis.set_ylabel("Clock angle [rad]")
            axis.set_zlabel("Acceleration angle")
            # axis.legend()
            axis.view_init(elev=90, azim=0)

        fig_1.colorbar(plt_1)
        fig_1.colorbar(plt_2)
        fig_1.colorbar(plt_3)
        fig_1.colorbar(plt_4)

        fig_1.tight_layout()
        fig_2.tight_layout()
        fig_3.tight_layout()
        fig_4.tight_layout()

        ax_4.set_zlabel("Angle")

        plt.show()


    def envelope_plots():

        import matplotlib

        SRP = Solar_pressure(sail_model="ACS3", central_attractor_mass=1.989 * 10 ** 30)
        state = [R_s * 10, 0, 0]

        def get_acc(tilt, clock):
            SRP.sail_control = [tilt, clock]
            acc = SRP.solar_acceleration(state=state)
            return acc

        alpha = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 100)
        gamma = np.linspace(0, 2 * math.pi, 100)

        tilt, clock = np.meshgrid(alpha, gamma)

        acc_mag = 0 * tilt
        x = 0 * tilt
        y = 0 * tilt

        for i, a in enumerate(alpha):
            for j, g in enumerate(gamma):
                acc_mag[i][j] = np.linalg.norm(get_acc(a, g))
                x[i][j] = np.sin(a) * np.sin(g)
                y[i][j] = np.sin(a) * np.cos(g)

        fig = plt.figure()
        ax_1 = fig.add_subplot(121, projection="3d")
        ax_1.plot_surface(tilt, clock, acc_mag, cmap=matplotlib.colormaps["jet"])

        ax_2 = fig.add_subplot(122, projection="3d")
        ax_2.plot_surface(x, y, acc_mag, cmap=matplotlib.colormaps["jet"])

        plt.show()

        pass


    def scatter_plots():

        import matplotlib

        SRP = Solar_pressure(sail_model="ACS3", central_attractor_mass=1.989 * 10 ** 30)
        state = [R_s * 10, R_s * 10, R_s * 10]

        def get_acc(tilt, clock):
            SRP.sail_control = [tilt, clock]
            acc = SRP.solar_acceleration(state=state)
            return acc

        alpha = np.linspace(0 * math.pi, 0.5 * math.pi, 30)
        gamma = np.linspace(0 * math.pi, 2 * math.pi, 30)

        [x, y, z] = [[] for _ in range(3)]

        for i, a in enumerate(alpha):
            for j, g in enumerate(gamma):
                acc = get_acc(a, g)
                x.append(acc[0])
                y.append(acc[1])
                z.append(acc[2])

        fig = plt.figure()
        ax_1 = fig.add_subplot(121, projection="3d")
        ax_1.scatter(x, y, z, color=[1, 0, 0], label="Positive tilt")

        """alpha = np.linspace(-0.5 * math.pi, 0 * math.pi, 50)
        gamma = np.linspace(0, math.pi, 50)

        [x, y, z] = [[] for _ in range(3)]

        for i, a in enumerate(alpha):
            for j, g in enumerate(gamma):
                acc = get_acc(a, g)
                x.append(acc[0])
                y.append(acc[1])
                z.append(acc[2])

        ax_1.scatter(x, y, z, color=[0, 1, 0], label="Negative alpha")"""

        ax_1.set_xlabel("X")
        ax_1.set_ylabel("Y")
        ax_1.set_zlabel("Z")
        # ax_1.set_zlim([-1, 1])
        lgnd = fig.legend()
        lgnd.set_draggable(True)
        plt.show()


    def curve_eval():
        SRP = Solar_pressure(sail_model="ACS3", central_attractor_mass=1.989 * 10 ** 30)
        state = [R_s * 10, 0, 0]

        def get_acc(tilt, clock):
            SRP.sail_control = [tilt, clock]
            acc = SRP.solar_acceleration(state=state)
            return acc

        tilt_angle = np.linspace(0 * math.pi, 0.49 * math.pi, 500)
        clock_angle = np.linspace(0 * math.pi, 0 * math.pi, 1)

        # acc_array = list(np.zeros(shape=(len(tilt_angle), len(clock_angle))))

        acc_x, acc_y, acc_z, TILT, CLOCK, ANGLE = (np.zeros(shape=(len(tilt_angle), len(clock_angle))) for _ in
                                                   range(6))

        for i, tilt in enumerate(tilt_angle):
            for j, clock in enumerate(clock_angle):
                acc_vec = get_acc(tilt, clock)
                TILT[i][j] = tilt * 180 / math.pi
                CLOCK[i][j] = clock * 180 / math.pi

                acc_x[i][j] = acc_vec[0]
                acc_y[i][j] = acc_vec[1]
                acc_z[i][j] = acc_vec[2]

                ANGLE[i][j] = math.acos((np.dot(acc_vec, np.array(state) - np.array(SRP.radiation_location))) / (
                        np.linalg.norm(np.array(acc_vec)) * np.linalg.norm(
                    np.array(state) - np.array(SRP.radiation_location))))

        fig = plt.figure()
        ax_1 = fig.add_subplot(121)
        ax_2 = fig.add_subplot(122)

        ax_1.plot(TILT, TILT - ANGLE * 180 / math.pi)
        ax_1.grid()

        ax_2.plot(TILT, ANGLE * 180 / math.pi)
        ax_2.grid()
        plt.show()


    cartesian_surface_plots()
