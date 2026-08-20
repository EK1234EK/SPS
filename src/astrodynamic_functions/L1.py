import numpy as np


def iter_step(state_b1: np.array, state_b2: np.array, r_l1, mu_1, mu_2):
    ang_vel = np.cross(state_b1[0:3], state_b1[3:6]) / (np.linalg.norm(r_l1)**2)
    v_l1 = np.cross(ang_vel, r_l1)
    cent_acc = (r_l1 / (np.linalg.norm(r_l1))) * (np.dot(r_l1.T, r_l1)) / (np.dot(r_l1.T, r_l1)) ** 0.5

    r_1 = r_l1 - state_b1[0:3]
    r_2 = r_l1 - state_b2[0:3]

    grav_acc_1 = - mu_1 / (np.dot(r_1.T, r_1) ** 1.5) * r_1
    grav_acc_2 = - mu_2 / (np.dot(r_2.T, r_2) ** 1.5) * r_2

    res_vec = cent_acc - grav_acc_1 - grav_acc_2
    return np.linalg.norm(res_vec)

def get_L1(state_b1: np.array, state_b2: np.array, mu_1, mu_2):
    r_samp = np.linspace(0.5*state_b2[0:3], 0.99*state_b2[0:3], 1000)
    res = [iter_step(state_b1, state_b2, r, mu_1, mu_2) for r in r_samp]
    pass