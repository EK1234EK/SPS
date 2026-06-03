import math
import numpy as np
from src.globals import Constants

G, _, KS_TOLERANCE, GRAV_CONST, EARTH_RADIUS = Constants.get_globals()


def Kepler_solver(M_e, e):
    while M_e < -2 * math.pi:
        M_e += 2 * math.pi
    error = 100
    a = -7
    b = 7
    n = 0
    m = -1
    while abs(error) > KS_TOLERANCE:
        m = (a + b) / 2
        f_m = m - e * math.sin(m) - M_e
        f_a = a - e * math.sin(a) - M_e

        if f_m * f_a < 0:
            b = m
        if f_m * f_a > 0:
            a = m

        error = f_m

        n += 1
        if n > 30:
            error = 0
    return m


def time_to_true_anomaly(a, e, ny_0, t, mass):
    MY = mass * GRAV_CONST
    n = (MY / (a ** 3)) ** (1 / 2)

    n = (MY / (a ** 3)) ** (1 / 2)
    E_0 = math.atan((math.tan(0.5 * ny_0) * ((1 - e) / (1 + e)) ** 0.5)) * 2
    M_0 = E_0 - e * math.sin(E_0)

    M = n * t + M_0
    while M > 2 * math.pi:
        M -= 2 * math.pi
    E = Kepler_solver(M, e)
    ny = 2 * math.atan(math.tan(0.5 * E) * ((1 + e) / (1 - e)) ** 0.5)

    if ny < 0:
        ny += 2 * math.pi
    return ny


def position_orbital_system(a, e, ny):
    p = a * (1 - e ** 2)
    r = p / (1 + e * math.cos(ny))

    x = r * math.cos(ny)
    y = r * math.sin(ny)
    z = 0
    return x, y, z


def rotate_orbit_to_inertial(INC, RAAN, APERI, vec):
    rot_APERI = [[math.cos(APERI), -math.sin(APERI), 0],
                 [math.sin(APERI), math.cos(APERI), 0],
                 [0, 0, 1]]

    rot_i = [[1, 0, 0],
             [0, math.cos(INC), -math.sin(INC)],
             [0, math.sin(INC), math.cos(INC)]]

    rot_RAAN = [[math.cos(RAAN), -math.sin(RAAN), 0], [math.sin(RAAN), math.cos(RAAN), 0], [0, 0, 1]]

    step_1 = np.matmul(rot_APERI, vec)
    step_2 = np.matmul(rot_i, step_1)
    step_3 = np.matmul(rot_RAAN, step_2)
    return step_3


def oe_to_sv(a, e, i, RAAN, APERI, ny_0, t, mass):
    # Additional arguments: Mass, in order to calculate the velocity vector. If no mass is provided,
    # only the position vector is calculated

    import warnings
    warnings.filterwarnings('ignore')

    """if len(args) == 0:
        ny = time_to_true_anomaly(a, e, ny_0, t)
        x, y, z = position_orbital_system(a, e, ny)
        vec_inertial = rotate_orbit_to_inertial(i, RAAN, APERI, [x, y, z])
        return vec_inertial[0], vec_inertial[1], vec_inertial[2]
    else:"""
    ny = time_to_true_anomaly(a, e, ny_0, t, mass)
    x, y, z = position_orbital_system(a, e, ny)
    vec_inertial = rotate_orbit_to_inertial(i, RAAN, APERI, [x, y, z])

    # Calculating the velocity vector:
    mu = mass * GRAV_CONST
    P = rotate_orbit_to_inertial(i, RAAN, APERI, [1, 0, 0])
    Q = rotate_orbit_to_inertial(i, RAAN, APERI, [0, 1, 0])

    # mag_vec = [-P[0] + Q[0], -P[1] + Q[1], -P[2] + Q[2]]

    p_mag = np.linalg.norm(vec_inertial) * (1 + e * math.cos(ny))

    v_vec = math.sqrt(mu / p_mag) * (-math.sin(ny) * P + Q * (e + math.cos(ny)))

    return [float(vec_inertial[0]), float(vec_inertial[1]), float(vec_inertial[2]), float(v_vec[0]),
            float(v_vec[1]), float(v_vec[2])]


def gravitational_law(mass, x, y, z):
    x_acc = -(GRAV_CONST * mass / (x ** 2 + y ** 2 + z ** 2)) * x / (x ** 2 + y ** 2 + z ** 2) ** 0.5
    y_acc = -(GRAV_CONST * mass / (x ** 2 + y ** 2 + z ** 2)) * y / (x ** 2 + y ** 2 + z ** 2) ** 0.5
    z_acc = -(GRAV_CONST * mass / (x ** 2 + y ** 2 + z ** 2)) * z / (x ** 2 + y ** 2 + z ** 2) ** 0.5

    return x_acc, y_acc, z_acc


def J_X_acceleration(mass, x, y, z):
    # Define the J-X constants:
    J_2 = 1082.63e-6
    J_3 = -2.53e-6
    J_4 = -1.61e-6

    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    my = GRAV_CONST * mass

    x_acc = (-my * x / (r ** 3)) * (1 -
                                    (1.5 * J_2 * (EARTH_RADIUS / r) ** 2) * (5 * ((z ** 2) / (r ** 2)) - 1) +
                                    (2.5 * J_3 * (EARTH_RADIUS / r) ** 3) * (3 * (z / r) - 7 * ((z ** 3) / (r ** 3))) -
                                    (5 / 8) * J_4 * (EARTH_RADIUS / r) ** 4 * (
                                            3 - 42 * ((z ** 2) / (r ** 2)) + 63 * ((z ** 4) / (r ** 4)))
                                    )
    y_acc = (y / x) * x_acc

    z_acc = (-my * z / (r ** 3)) * (1 +
                                    (1.5 * J_2 * (EARTH_RADIUS / r) ** 2) * (3 - 5 * ((z ** 2) / (r ** 2))) +
                                    (1.5 * J_3 * (EARTH_RADIUS / r) ** 3) * (
                                            10 * (z / r) - (35 / 3) * ((z ** 3) / (r ** 3)) - z / r) -
                                    (5 / 8) * J_4 * (EARTH_RADIUS / r) ** 4 * (
                                            15 - 70 * ((z ** 2) / (r ** 2)) + 63 * ((z ** 4) / (r ** 4)))
                                    )

    return x_acc, y_acc, z_acc


def CR3BP_acceleration(x, y, z, vx, vy, my):
    r_1 = ((x + my) ** 2 + y ** 2 + z ** 2) ** 0.5
    r_2 = ((x - (1 - my)) ** 2 + y ** 2 + z ** 2) ** 0.5

    x_acc = 2 * vy + x - ((1 - my) * (x + my)) / (r_1 ** 3) - (my * (x - (1 - my))) / (r_2 ** 3)

    y_acc = -2 * vx + y - ((1 - my) * y) / (r_1 ** 3) - (my * y) / (r_2 ** 3)

    z_acc = - ((1 - my) * z) / (r_1 ** 3) - (my * z) / (r_2 ** 3)
    return x_acc, y_acc, z_acc


def sv_to_oe(state_vector, mass, *args):
    import warnings
    warnings.filterwarnings('ignore')

    r = np.array(state_vector[0:3])
    v = np.array(state_vector[3:6])

    mu = mass * GRAV_CONST
    mag_r = np.sqrt(r.dot(r))
    mag_v = np.sqrt(v.dot(v))

    h = np.cross(r, v)
    mag_h = np.linalg.norm(h)
    # mag_h = np.sqrt(h.dot(h))

    # e = ((np.cross(v, h)) / mu) - (r / mag_r)
    e = (1 / mu) * ((mag_v ** 2 - mu / mag_r) * r - r.dot(v) * v)
    mag_e = np.linalg.norm(e)
    # mag_e = np.sqrt(e.dot(e))

    n = np.array([-h[1], h[0], 0])
    mag_n = np.linalg.norm(n)
    # mag_n = np.sqrt(n.dot(n))

    true_anom = math.acos(np.clip(np.dot(e, r) / (mag_r * mag_e), -1, 1))
    if np.dot(r, v) < 0:
        true_anom = 2 * math.pi - true_anom
    # true_anom = math.degrees(true_anom)

    i = math.acos(np.clip(h[2] / mag_h, -1, 1))
    # i = math.degrees(i)

    ecc = mag_e

    raan = math.acos(np.clip(n[0] / mag_n, -1, 1))
    if n[1] < 0:
        raan = 2 * math.pi - raan
    # raan = math.degrees(raan)

    per = math.acos(np.clip(np.dot(n, e) / (mag_n * mag_e), -1, 1))
    if e[2] < 0:
        per = 2 * math.pi - per
    # per = math.degrees(per)

    a = 1 / ((2 / mag_r) - (mag_v ** 2 / mu))

    if i >= 2 * math.pi:
        i = i - 2 * math.pi
    if raan >= 2 * math.pi:
        raan = raan - 2 * math.pi
    if per >= 2 * math.pi:
        per = per - 2 * math.pi

    if len(args) != 0:
        # Calculate true anomaly at epoch from true anomaly
        true_anom = time_to_true_anomaly(a, ecc, true_anom,args[0], mass)

    kep = np.zeros(6)
    kep[0] = a
    kep[1] = ecc
    kep[2] = i
    kep[3] = raan
    kep[4] = per
    kep[5] = true_anom
    return kep


def get_sv_list(sc_list, idx):
    sv_list = [[] for _ in range(len(sc_list))]
    for i, sc in enumerate(sc_list):
        sv_list[i] = [sc.trajectory_track[k][idx] for k in range(6)]
    return sv_list


"""if __name__ == "__main__":
    tspan = np.linspace(0, -1e8, 100).tolist()
    for t in tspan:
        veci = oe_to_sv(1e11, 0.1, 0, 0, 0, 0, t)
        vec = [float(veci[0]), float(veci[1]), float(veci[2])]
        # print(t)"""
