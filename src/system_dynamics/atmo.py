import math
import numpy as np
from src.globals.Constants import get_globals
G, MY, KS_TOLERANCE, GRAV_CONST, EARTH_RADIUS = get_globals()


def get_coeff():
    coeff = {0: [7.249, 1.225E+00],
    25000: [6.349, 3.899E-02],
    30000: [6.682, 1.774E-02],
    40000: [7.554, 3.972E-03],
    50000: [8.382, 1.057E-03],
    60000: [7.714, 3.206E-03],
    70000: [6.549, 8.770E-05],
    80000: [5.799, 1.905E-05],
    90000: [5.382, 3.396E-06],
    100000: [5.877, 5.297E-07],
    110000: [7.263, 9.661E-08],
    120000: [9.473, 2.438E-08],
    130000: [12.636, 8.484E-09],
    140000: [16.149 ,3.845E-09],
    150000: [22.523, 2.070E-09],
    180000: [29.74, 5.464E-10],
    200000: [37.105, 2.789E-10],
    250000: [45.546, 7.248E-11],
    300000: [53.628, 2.418E-11],
    350000: [53.298, 9.518E-12],
    400000: [58.515, 3.725E-12],
    450000: [60.828, 1.585E-12],
    500000: [63.822, 6.967E-13],
    600000: [71.835, 1.454E-13],
    700000: [88.667, 3.614E-14],
    800000: [124.64, 1.170E-14],
    900000: [81.05, 5.245E-15],
    1000000: [100, 0]}
    return coeff

class Atmopshere:
    def __init__(self):
        self.coeff = get_coeff()  # On the form {init_altitude: [scale_height, density]}
        self.H = None
        self.scale_rho = None
        self.lower_altitude = None
        self.rho = None
        self.keys = list(self.coeff.keys())

    def get_params(self, height):
        key = self.keys[-1]
        for i, key in enumerate(self.keys):
            if height > key:
                self.lower_altitude = self.keys[i-1]
                break
        params = self.coeff[key]
        self.H = params[0]
        self.rho = params[1]

    def get_density(self, state):
        h = np.linalg.norm(state[0:3]) - EARTH_RADIUS
        self.rho = self.scale_rho * math.exp(-(h - self.lower_altitude) / self.H)
