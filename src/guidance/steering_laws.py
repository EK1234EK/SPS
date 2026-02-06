from scipy.optimize import direct

from src.astrodynamic_functions import kepler_dynamics
import numpy as np


class LocalOptimal:
    def __init__(self):
        self.jacobian = [[]]
        self.target = ""
        self.conversion_mass = -1

        self.current_control = []

        self.oe_direction = []
        self.target_oe = dict()


    def maximize_oe_change(self, state):
        pass
        """Targets the current acceleration vector in such a way that the change in a certain
        orbital parameter becomes maximal"""

        """To start off, we obtain the Jacobian of the orbital parameters with respect
        to the change in the velocity vector """

        dv = 0.1  # Incement, by which to estimate the derivativs in the Jacobian
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

    def maximize_oe_set(self, state):
        pass
        """Targets the current acceleration vector in such a way that the change in a certain
        orbital parameter becomes maximal"""

        """To start off, we obtain the Jacobian of the orbital parameters with respect
        to the change in the velocity vector """

        if not self.oe_direction:
            raise ValueError("No target orbital parameters defined!")

        oe_direction = np.array(self.oe_direction)

        dv = 0.1  # Incement, by which to estimate the derivativs in the Jacobian
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

        J = np.array(jacobian)

        J_T_p = np.linalg.pinv(J)

        dr = J_T_p.dot(oe_direction)
        mag = (dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2) ** 0.5
        if mag == 0:
            return [0, 0, 0]
        direction = dr / mag
        direction = list(direction * acc_mag)
        return direction

        pass


    def target_orbit(self, state):

        if self.target_oe == {}:
            raise ValueError("No target orbital parameter set defined!")

        oe_name_list = ["SMA", "ECC", "INC", "RAAN", "APERI", "TAEPO"]

        dv = 0.1  # Incement, by which to estimate the derivativs in the Jacobian
        acc_mag = 0.0005

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

        # Filling in the empty parameters in the taret orbital element list
        target_oe_list = [0 for _ in range(6)]
        for i, oe in enumerate(oe_name_list):
            if oe in self.target_oe.keys():
                target_oe_list[i] = self.target_oe[oe] - oe_mat["center"][i]
            else:
                target_oe_list[i] = 0

        target_oe_list = np.array(target_oe_list)

        jacobian = [[0, 0, 0] for _ in range(6)]

        for i in range(6):
            for j, delta_state in enumerate(["+vx", "+vy", "+vz"]):
                # We are now using the difference between target and current orbital elements for the Jacobian
                jacobian[i][j] = (oe_mat[delta_state][i] - oe_mat["center"][i]) / dv

        J = np.array(jacobian)

        J_T_p = np.linalg.pinv(J)

        dr = J_T_p.dot(target_oe_list)
        mag = (dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2) ** 0.5
        if mag == 0:
            return [0, 0, 0]
        direction = dr / mag
        direction = list(direction * acc_mag)
        self.current_control = direction

    def guidance(self, state, time, force_model):
        if time < 15000000:
            self.target_oe = {"SMA": 200000000, "ECC": 0.1, "INC": 0.1, "RAAN": 1.0, "APERI": 1}
        elif 15000000 < time < 30000000:
            self.target_oe = {"SMA": 400000000, "ECC": 0.5, "INC": 0.6, "RAAN": 2.0, "APERI": 1}
        elif 30000000 < time < 40000000:
            self.target_oe = {"SMA": 300000000, "ECC": 0.1, "INC": 1.1, "RAAN": 4.0, "APERI": 1}
        elif 40000000 < time < 500000000:
            self.target_oe = {"SMA": 400000000, "ECC": 0.05, "INC": 0.2, "RAAN": 4.0, "APERI": 1}

        self.target_orbit(state=state)
        return self.current_control