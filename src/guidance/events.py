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


def kill_integrator_SMA_INC(time, state, *args):
    OP = kepler_dynamics.sv_to_oe(state_vector=state, mass=5.97e24)
    SMA = OP[0]
    INC = OP[2]
    if 0.98 * args[0] < SMA < 1.02 * args[0] and 26*math.pi/180 < INC < 30*math.pi/180:
        return 1
    else:
        return -1

def kill_integrator_SMA(time, state, cutoff):
    OP = kepler_dynamics.sv_to_oe(state_vector=state, mass=5.97e24)
    SMA = OP[0]
    if cutoff < SMA:
        return 1
    else:
        return -1
    """print(round(float(args[0] - SMA)), "   " , end="")
    if args[0] < SMA:
        print(1)
        return 1
    else:
        print(-1)
        return -1"""

def kill_integrator_interface(time, state, cutoff):
    oe = kepler_dynamics.sv_to_oe(state_vector=state, mass=5.9722e24)
    # if not (49877718.58332233 < oe[0] < 199590371.3654217):
    if not (49877718.58332233 < oe[0] < 179590371.3654217):
        return -1
    if not (0.1294540679910744 < oe[1] < 0.4927204547048896):
        return -1
    if not (0.0792983383511437 < oe[2] < 0.6051461805592163):
        return -1
    return 1


def kill_integrator_interface_convex(time, state, interface_inst):
    oe = kepler_dynamics.sv_to_oe(state_vector=state, mass=5.9722e24)
    interp_val = interface_inst.interface_interpolation(eval_points=[oe[0], oe[1], oe[2]], fill_val=69)
    if interp_val != 69:
        return -1
    else:
        return 1