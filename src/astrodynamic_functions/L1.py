import numpy as np
import matplotlib.pyplot as plt

import src.globals.Constants

G, MY, KS_TOLERANCE, GRAV_CONST, EARTH_RADIUS, OBLIQUITY = src.globals.Constants.get_globals()


def iter_step(state_b1: np.array, state_b2: np.array, r_l1, mu_1, mu_2):
    ang_vel = np.cross(state_b1[0:3], state_b1[3:6]) / (np.linalg.norm(r_l1)**2)
    v_l1 = np.cross(ang_vel, r_l1)
    cent_acc = (r_l1 / (np.linalg.norm(r_l1))) * (np.dot(v_l1.T, v_l1)) / ((np.dot(r_l1.T, r_l1)) ** 0.5)

    r_1 = r_l1 - state_b1[0:3]
    r_2 = r_l1 - state_b2[0:3]

    grav_acc_1 = - (mu_1 / (np.dot(r_1.T, r_1) ** 1.5)) * r_1
    grav_acc_2 = - (mu_2 / (np.dot(r_2.T, r_2) ** 1.5)) * r_2

    res_vec = +cent_acc - grav_acc_1 - grav_acc_2
    return np.linalg.norm(res_vec)

def get_L1(state_b1: np.array, state_b2: np.array, mu_1, mu_2, nsamp):
    x_samp = np.linspace(0.5 * state_b2[0], 0.99 * state_b2[0], nsamp)
    y_samp = np.linspace(0.5 * state_b2[1], 0.99 * state_b2[1], nsamp)
    z_samp = np.linspace(0.5 * state_b2[2], 0.99 * state_b2[2], nsamp)
    res = [abs(iter_step(state_b1, state_b2, np.array([x_samp[i], y_samp[i], z_samp[i]]), mu_1, mu_2)) for i in range(nsamp)]

    idx = np.argmin(res)

    """fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, len(res)), res)
    plt.show()"""
    return np.array([x_samp[idx], y_samp[idx], z_samp[idx]])

if __name__ == "__main__":
    state_b1 = [[-4622952.803546128], [0], [0.0], [0], [-12.512944849494628], [0.0]]
    state_b2 = [[375940935.134163], [0.0], [0.0], [0.0], [1017.1847091494611], [0.0]]
    state_b1 = [state_b1[i][0] for i in range(6)]
    state_b2 = [state_b2[i][0] for i in range(6)]
    mu_1 = 5.9722e24 * GRAV_CONST
    mu_2 = 7.347e22 * GRAV_CONST

    n_samp = 100000
    L1 = get_L1(state_b1, state_b2, mu_1, mu_2, n_samp)
    pass
