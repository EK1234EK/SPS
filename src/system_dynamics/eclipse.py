import numpy as np
import math

from astropy.coordinates.builtin_frames.ecliptic_transforms import ecliptic_to_iau76_icrs
from numpy.ma.core import arccos

import src.globals.Constants

sigma_star, L_s, R_s, c  = src.globals.Constants.get_SRP_globals()

class eclipse_model:
    def __init__(self, eclipse_bodies: dict, force_model):
        # Use "central attractor" when passing the exlipse_body array, then this body is assumed at [0, 0, 0]
        # eclipse_bodies = {"body_name": body_dia, ...}
        self.eclipsing_bodies = eclipse_bodies
        self.force_model = force_model


    def get_single_body_eclipse(self, body_name, body_dia, time, sc_pos: np.array, sun_pos: np.array):
        if body_name == "central_attractor":
            s = sc_pos
        else:
            body_states = self.force_model.propagate_body_states(times=[time], mass=self.force_model.central_mass, body_list=[body_name])[body_name]
            r_b = np.array(body_states[0:3])
            s = sc_pos - r_b

        a = math.asin(R_s / np.linalg.norm(sun_pos - sc_pos))
        b = math.asin(body_dia / np.linalg.norm(s))
        c = math.asin((-np.transpose(s) * (sun_pos - sc_pos)) / (np.linalg.norm(sun_pos - sc_pos)))

        x = (c**2 + a**2 - b**2) / (2 * c)
        y = (a**2 - x**2)**0.5

        psi = a**2 * math.acos(x / a) + b**2 * math.acos((c - x) / b) - c * y

        nu = 1 - psi / (math.pi * a**2)
        return nu

    def get_eclipse_factor(self, sc_pos, sun_pos, time):
        nu = 1
        for i, key in enumerate(self.eclipsing_bodies.keys()):
            nu = nu * self.get_single_body_eclipse(body_name=key, body_dia=self.eclipsing_bodies[key]["dia"], time=time, sc_pos=sc_pos, sun_pos=sun_pos)
        return nu