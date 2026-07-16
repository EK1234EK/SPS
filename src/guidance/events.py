import math

from src.astrodynamic_functions import kepler_dynamics
from src.astrodynamic_functions.kepler_dynamics import GRAV_CONST, EARTH_RADIUS
import numpy as np


def kill_integrator_eccentricity(time, state):
    return kepler_dynamics.sv_to_oe(state_vector=state, mass=5.97e24)[1] - 0.99


def kill_integrator_C3(time, state):
    SMA = kepler_dynamics.sv_to_oe(state_vector=state, mass=5.97e24)[0]
    C3 = -5.97e24 * GRAV_CONST / SMA + 100000
    return C3


def kill_integrator_SMA(time, state):
    OP = kepler_dynamics.sv_to_oe(state_vector=state, mass=5.97e24)
    SMA = OP[0]
    INC = OP[2]
    if 9000000 < SMA < 11000000 and 9 * math.pi / 180 < INC < 11 * math.pi / 180:
        return 1
    else:
        return -1


def kill_integrator_altitude(time, state):
    radius = np.linalg.norm(np.array(state[0:3]))
    return radius - 100000000