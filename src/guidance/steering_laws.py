from src.astrodynamic_functions import kepler_dynamics
import numpy as np


class LocalOptimal:
    def __init__(self):
        self.jacobian = [[]]
        self.target = ""
        self.conversion_mass = -1

    def maximize_oe_change(self, state):
        pass
        """Targets the current acceleration vector in such a way that the change in a certain
        orbital parameter becomes maximal"""

        """To start off, we obtain the Jacobian of the orbital parameters with respect
        to the change in the velocity vector """

        dv = 10  # Incement, by which to estimate the derivativs in the Jacobian
        acc_mag = 0.001

        # The Jacobian has six lines with three columns, corresponding to six orbital parameters
        # and the three velocity state

        state_mat = dict()
        oe_mat = dict()
        state_mat["center"] = state
        oe_mat["center"] = kepler_dynamics.sv_to_oe(state_vector=state_mat["center"], mass=self.conversion_mass)

        dvx = [0, 0, 0, dv, 0, 0]
        state_mat["+vx"] = [state[i] + dvx[i] for i in range(6)]
        oe_mat["+vx"] = kepler_dynamics.sv_to_oe(state_vector=state_mat["+vx"], mass=self.conversion_mass)

        dvy = [0, 0, 0, 0, dv, 0]
        state_mat["+vy"] = [state[i] + dvy[i] for i in range(6)]
        oe_mat["+vy"] = kepler_dynamics.sv_to_oe(state_vector=state_mat["+vy"], mass=self.conversion_mass)

        dvz = [0, 0, 0, 0, 0, dv]
        state_mat["+vz"] = [state[i] + dvz[i] for i in range(6)]
        oe_mat["+vz"] = kepler_dynamics.sv_to_oe(state_vector=state_mat["+vz"], mass=self.conversion_mass)

        jacobian = [[0, 0, 0] for _ in range(6)]

        for i in range(6):
            for j, delta_state in enumerate(["+vx", "+vy", "+vz"]):
                jacobian[i][j] = (oe_mat[delta_state][i] - oe_mat["center"][i]) / dv

        oe_list = ["SMA", "ECC", "INC", "RAAN", "APERI", "TAEPO"]

        if self.target in oe_list:
            idx = oe_list.index(self.target)
            direction = np.array(jacobian[idx][:])
            mag = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
            if mag == 0:
                return [0, 0, 0]
            direction = direction / mag
            direction = list(direction * acc_mag)

            return direction

        else:
            raise ValueError("Steering target not in list of known targets")
