import numpy as np
import math

def sail_attitude(sail_control, radiation_location):
    alpha = sail_control[0]  # Tilt angle
    gamma = sail_control[1]  # Clock angle

    d_1 = np.array([state[i] - radiation_location[i] for i in range(3)])
    d_1 = d_1 / np.linalg.norm(d_1)

    d_2 = np.cross(d_1, np.array([0, 0, 1]))  # I'm sure this will lead to problems later on
    d_2 = d_2 / np.linalg.norm(d_2)

    d_3 = np.cross(d_2, d_1)
    d_3 = d_3 / np.linalg.norm(d_3)

    n = math.cos(alpha) * d_1 + math.sin(alpha) * math.sin(gamma) * d_2 + math.sin(alpha) * math.cos(gamma) * d_3
    return d_1, d_2, d_3, n

class Solar_pressure:

    def __init__(self, base_acc = 1.7 * 10 ** (-5)):
        self.base_acc = base_acc
        self.radiation_location = [0, 0, 0]

        self.sail_control = [0, 0]


    def ideal_sail(self, radiation_location, state):

        """alpha = self.sail_control[0]  # Tilt angle
        gamma = self.sail_control[1]  # Clock angle

        d_1 = np.array([state[i] - radiation_location[i] for i in range(3)])
        d_1 = d_1 / np.linalg.norm(d_1)

        d_2 = np.cross(d_1, np.array([0, 0, 1]))  # I'm sure this will lead to problems later on
        d_2 = d_2 / np.linalg.norm(d_2)

        d_3 = np.cross(d_2, d_1)
        d_3 = d_3 / np.linalg.norm(d_3)

        n = math.cos(alpha) * d_1 + math.sin(alpha) * math.sin(gamma) * d_2 + math.sin(alpha) * math.cos(gamma) * d_3"""

        d_1, d_2, d_3, n = sail_attitude(sail_control=self.sail_control, radiation_location=self.radiation_location)

        # Taking into account the direction of the resulting acceleration:

        if np.dot(d_1, n) > 0:
            acc = list(self.base_acc * (np.dot(-d_1, n))**2 * n)
        else:
            acc = list(-self.base_acc * (np.dot(-d_1, n)) ** 2 * n)
        return acc

    def real_sail(self, radiation_location, state):

        d_1, d_2, d_3, n = sail_attitude(sail_control=self.sail_control, radiation_location=self.radiation_location)

        # Determining the t-vector

        return [0, 0, 0]

    def solar_acceleration(self, state,):
        return self.ideal_sail(self.radiation_location, state)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'

    SRP = Solar_pressure(base_acc=1)
    SRP.sail_control = [0, 0.4]
    state= [1, 0, 0]

    def get_acc(tilt, clock):
        SRP.sail_control = [tilt, clock]
        acc = SRP.solar_acceleration(state=state)
        return acc

    tilt_angle = np.linspace(-math.pi, math.pi, 50)
    clock_angle = np.linspace(-math.pi, math.pi, 50)

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

            ANGLE[i][j] = math.acos((np.dot(acc_vec, np.array(state) - np.array(SRP.radiation_location))) / (np.linalg.norm(np.array(acc_vec)) * np.linalg.norm(np.array(state) - np.array(SRP.radiation_location))))


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

