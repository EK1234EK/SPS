from csv import excel

import numpy as np
import math

from astropy.coordinates.builtin_frames.ecliptic_transforms import ecliptic_to_iau76_icrs
from numba.cuda.testing import skip_with_cuda_python
from numpy.ma.core import arccos
from pandas.io.sas.sas_constants import subheader_pointers_offset

import src.globals.Constants

sigma_star, L_s, R_s, c  = src.globals.Constants.get_SRP_globals()

class eclipse_model:
    def __init__(self, eclipse_bodies: dict, force_model):
        # Use "central attractor" when passing the exlipse_body array, then this body is assumed at [0, 0, 0]
        # eclipse_bodies = {"body_name": body_dia, ...}
        self.eclipsing_bodies = eclipse_bodies
        self.force_model = force_model


    def get_single_body_eclipse(self, body_name, body_radius, time, sc_pos: np.array, sun_pos: np.array):
        if body_name == "central_attractor":
            s = sc_pos
        else:
            body_states = self.force_model.propagate_body_states(times=[time], mass=self.force_model.central_mass, body_list=[body_name])[body_name]
            r_b = np.array(body_states[0:3])
            s = sc_pos - r_b

        a = math.asin(R_s / np.linalg.norm(sun_pos - sc_pos))
        b = math.asin(body_radius / np.linalg.norm(s))

        c = math.acos(np.dot(-np.transpose(s), (sun_pos - sc_pos)) / (np.linalg.norm(s) * np.linalg.norm(sun_pos - sc_pos)))  # TODO this is acos according to source Montenbruck_2000_SatelliteOrbits
        # c = math.asin(np.dot(-np.transpose(s), (sun_pos - sc_pos)) / (np.linalg.norm(s) * np.linalg.norm(sun_pos - sc_pos)))

        # Check, if the eclipse condition is met
        if abs(a - b) < c < (a + b):
            # print("Transition: |a-b|: ", abs(a - b), "; c: ", c, "; a+b: ", a + b)
            x = (c ** 2 + a ** 2 - b ** 2) / (2 * c)
            y = (a ** 2 - x ** 2) ** 0.5

            psi = a ** 2 * math.acos(x / a) + b ** 2 * math.acos((c - x) / b) - c * y

            nu = 1 - (psi / (math.pi * a ** 2))
            """elif abs(a - b) > c:
            print("Full eclipse: |a-b|: ", abs(a - b), "; c: ", c, "; a+b: ", a + b)
            nu = 0"""
        elif abs(a - b) > c:
            c = abs(a - b) + 0.0000001
            # print("Intermediate: |a-b|: ", abs(a - b), "; c: ", c, "; a+b: ", a + b)
            x = (c ** 2 + a ** 2 - b ** 2) / (2 * c)
            y = (a ** 2 - x ** 2) ** 0.5

            psi = a ** 2 * math.acos(x / a) + b ** 2 * math.acos((c - x) / b) - c * y

            nu = 1 - (psi / (math.pi * a ** 2))
            """elif abs(a - b) > c:
            print("Full eclipse: |a-b|: ", abs(a - b), "; c: ", c, "; a+b: ", a + b)
            nu = 0"""

        else:
            # print("No shadow: |a-b|: ", abs(a - b), "; c: ", c, "; a+b: ", a + b)
            nu = 1

        # print(nu)
        return nu

    def get_eclipse_factor(self, sc_pos, sun_pos, time):
        nu = 1
        for i, key in enumerate(self.eclipsing_bodies.keys()):
            nu = nu * self.get_single_body_eclipse(body_name=key, body_radius=self.eclipsing_bodies[key], time=time, sc_pos=np.array(sc_pos), sun_pos=sun_pos)
        return nu

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    body_name = "central_attractor"
    R_b = 6378000
    time = None
    d = R_b * 230
    d_s = 1.49 * 10**11
    sun_pos = d_s * np.array([-1, 0, 0])

    Interface = eclipse_model(None, None)

    phi_arr = np.linspace(-0.1, 0.1, 10000)
    nu_arr = 0 * phi_arr

    for i, phi in enumerate(phi_arr):
        sc_pos = d * np.array([math.cos(phi), math.sin(phi), 0])
        nu = Interface.get_single_body_eclipse(body_name=body_name, body_radius=R_b, time=time, sc_pos=sc_pos, sun_pos=sun_pos)
        nu_arr[i] = nu

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(phi_arr, nu_arr)
    plt.show()

