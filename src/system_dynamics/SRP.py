import numpy as np
import math

class Solar_pressure:

    def __init__(self, base_acc = 1.7 * 10 ** (-5)):
        self.base_acc = base_acc
        self.radiation_location = [0, 0, 0]

        self.sail_control = [0, 0]


    def model_1(self, radiation_location, state):

        alpha = self.sail_control[0]  # Tilt angle
        gamma = self.sail_control[1]  # Clock angle

        d_1 = np.array([state[i] - radiation_location[i] for i in range(3)])
        d_1 = d_1 / np.linalg.norm(d_1)

        d_2 = np.cross(d_1, np.array([0, 0, 1]))  # I'm sure this will lead to problems later on
        d_2 = d_2 / np.linalg.norm(d_2)

        d_3 = np.cross(d_2, d_1)
        d_3 = d_3 / np.linalg.norm(d_3)

        n = math.cos(alpha) * d_1 + math.sin(alpha) * math.sin(gamma) * d_2 + math.sin(alpha) * math.cos(gamma) * d_3

        # Taking into account the direction of the resulting acceleration:

        if np.dot(d_1, n) > 0:
            acc = list(self.base_acc * (np.dot(-d_1, n))**2 * n)
        else:
            acc = list(-self.base_acc * (np.dot(-d_1, n)) ** 2 * n)
        return acc

    def solar_acceleration(self, state,):
        return self.model_1(self.radiation_location, state)