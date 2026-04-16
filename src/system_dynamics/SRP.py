import numpy as np
import math


def sail_attitude(sail_control, radiation_location, state):
    alpha = sail_control[0]  # Tilt angle
    gamma = sail_control[1]  # Clock angle

    d_1 = np.array(state) - np.array(radiation_location)
    d_1 = d_1 / np.linalg.norm(d_1)

    d_2 = np.cross(d_1, np.array([0, 0, 1]))  # I'm sure this will lead to problems later on
    d_2 = d_2 / np.linalg.norm(d_2)

    d_3 = np.cross(d_2, d_1)
    d_3 = d_3 / np.linalg.norm(d_3)

    n = math.cos(alpha) * d_1 + math.sin(alpha) * math.sin(gamma) * d_2 + math.sin(alpha) * math.cos(gamma) * d_3
    return d_1, d_2, d_3, n


class Solar_pressure:

    def __init__(self, base_acc=1.7 * 10 ** (-5)):
        self.base_acc = base_acc
        self.radiation_location = [0, 0, 0]

        self.sail_control = [0, 0]

    def ideal_sail(self, state):

        d_1, d_2, d_3, n = sail_attitude(sail_control=self.sail_control, radiation_location=self.radiation_location, state=state)

        # Taking into account the direction of the resulting acceleration:

        if np.dot(d_1, n) > 0:
            acc = list(self.base_acc * (np.dot(-d_1, n)) ** 2 * n)
        else:
            acc = list(-self.base_acc * (np.dot(-d_1, n)) ** 2 * n)
        return acc

    def real_sail(self, radiation_location, state):

        alpha = self.sail_control[0]
        clock = self.sail_control[1]

        # Parameter defintions:
        P = 1  # Photonic pressure
        A = 1  # Sail area
        r = 0.9  # Fraction of incidence photons that is reflectd
        s = 0.82  # Fraction of photons that experiences specular reflection
        B_f = 0.79  # Scattered fraction, frontside
        B_b = 0.67  # Scattered fraction, backside

        epsilon_f = 0.03  # Emissivity front
        epsilon_B = 0.6  # Emissivity back

        m = 1  # Mass of the spacecraft

        d_1, d_2, d_3, n = sail_attitude(sail_control=self.sail_control, radiation_location=self.radiation_location)

        """if np.dot(d_1, n) < 0:
            n *= -1"""

        # Determining the t-vector

        incidence = np.array(state[0:3]) - np.array(self.radiation_location)
        temp = np.cross(n, incidence)
        t = np.cross(temp, n)
        t = t / np.linalg.norm(t)

        # Determining the two pre-factors:
        # Normal force
        f_n = P * A * ((1 + r * s) * math.cos(alpha) ** 2 + B_f * (1 - s) * r * math.cos(alpha) + (1 - r) * (
                    epsilon_f * B_f - epsilon_B - B_b) / (epsilon_f + epsilon_B)) * n

        f_t = P * A * (1 - r*s)*math.cos(alpha)*math.sin(alpha) * t

        acc = (f_n + f_t) / m
        return acc

    def solar_acceleration(self, state):
        return self.ideal_sail(state=state)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'

    SRP = Solar_pressure(base_acc=1)
    state = [1, 0, 0]


    def get_acc(tilt, clock):
        SRP.sail_control = [tilt, clock]
        acc = SRP.solar_acceleration(state=state)
        return acc


    tilt_angle = np.linspace(-math.pi*0, math.pi*0, 50)
    clock_angle = np.linspace(-math.pi*0, math.pi*0, 50)

    # acc_array = list(np.zeros(shape=(len(tilt_angle), len(clock_angle))))

    acc_x, acc_y, acc_z, TILT, CLOCK, ANGLE = (np.zeros(shape=(len(tilt_angle), len(clock_angle))) for _ in range(6))

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
    plt_4 = ax_4.plot_surface(TILT, CLOCK, ANGLE * 180 / math.pi, label="Acceleration angle", cmap=mpl.colormaps["hsv"])

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
