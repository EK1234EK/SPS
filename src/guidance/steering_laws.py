from scipy.optimize import direct
from scipy.stats import alpha

import src.system_dynamics.SRP
from src.astrodynamic_functions import kepler_dynamics
import numpy as np
import math
from src.system_dynamics import SRP


def kill_integrator(time, state):
    return kepler_dynamics.sv_to_oe(state_vector=state, mass=5.97e24)[1] - 0.95


def vector_from_angle(alpha, gamma, d_1, d_2, d_3):
    return math.cos(alpha) * d_1 + math.sin(alpha) * math.sin(gamma) * d_2 + math.sin(alpha) * math.cos(gamma) * d_3


def angle_from_vector(v, d_1, d_2, d_3):
    alpha = math.acos((v.dot(d_1)) / (np.linalg.norm(v) * np.linalg.norm(d_1)))

    # v_d1 = (d_1 / (np.linalg.norm(d_1)) ** 2) * v.dot(d_1)  # Projection onto d_1
    # v_plane = v - v_d1  # Component in d_2 - d_3 plane
    # gamma = math.acos((d_3.dot(v_plane)) / (np.linalg.norm(d_3) * np.linalg.norm(v_plane)))

    v_d2 = ((d_2 / (np.linalg.norm(d_2)) ** 2) * v.dot(d_2)).dot(d_2 / (np.linalg.norm(d_2)))  # Projection onto d_2
    v_d3 = ((d_3 / (np.linalg.norm(d_3)) ** 2) * v.dot(d_3)).dot(d_3 / (np.linalg.norm(d_3))) # Projection onto d_3
    gamma = math.atan2(v_d2, v_d3)
    if gamma < 0: gamma += 2*math.pi

    return alpha, gamma


def direct_control_inversion(vel_change, state, force_model):
    # Attention only use with ideal sail!
    d_1, d_2, d_3, n = src.system_dynamics.SRP.sail_attitude([0, 0],
                                                             radiation_location=force_model.solar_pressure.radiation_location,
                                                             state=state[0:3])
    alpha_v, gamma_v = angle_from_vector(vel_change, d_1, d_2, d_3)

    if np.dot(n, vel_change) < 0:
        alpha = 0.5*math.pi
        gamma = gamma_v
        return [alpha, gamma], [alpha_v, gamma_v]
    gamma = gamma_v
    alpha = math.atan((-3 + (9 + 8 * math.tan(alpha_v)**2)**0.5) / (4 * math.tan(alpha_v)))
    if alpha < 0:
        alpha *= -1


    return [alpha, gamma], [alpha_v, gamma_v]


class LocalOptimal:
    def __init__(self):
        self.jacobian = [[]]
        self.target = ""
        self.conversion_mass = -1

        self.current_control = []

        self.oe_direction = []
        self.target_oe = {"True time": []}

        self.track_time = 0

        self.control_command_track = dict()
        self.vel_angle_track = dict()

        self.prev_time = 0
        self.prev_acc = np.array([0, 0, 0])
        self.scaling_acc = 0

        self.terminate_integration = -1 # Set to zero to terminate!

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

    def target_orbit(self, state):

        if self.target_oe == {}:
            raise ValueError("No target orbital parameter set defined!")

        oe_name_list = ["SMA", "ECC", "INC", "RAAN", "APERI", "TAEPO"]

        dv = 0.1  # Incement, by which to estimate the derivativs in the Jacobian
        acc_mag = 1

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
            self.current_control = [0, 0, 0]
            return None

        direction = dr / mag
        direction = list(direction * acc_mag)
        # self.current_control = direction
        return direction


    def target_inline_lagrange(self, state):

        if self.target_oe == {}:
            raise ValueError("No target orbital parameter set defined!")

        oe_name_list = ["SMA", "ECC", "INC", "Lon"]

        dv = 0.1  # Incement, by which to estimate the derivativs in the Jacobian
        acc_mag = 0.005

        # The Jacobian has six lines with three columns, corresponding to six orbital parameters
        # and the three velocity state

        state_mat = dict()
        oe_mat = dict()
        solution_mat = dict()
        state_mat["center"] = state
        oe_mat["center"] = kepler_dynamics.sv_to_oe(state_vector=state_mat["center"], mass=self.conversion_mass)
        solution_mat["center"] = oe_mat["center"][0:2] + [
            oe_mat["center"][3] + oe_mat["center"][4] + oe_mat["center"][5]]

        dvx = [0, 0, 0, dv, 0, 0]
        state_mat["+vx"] = [state[i] + dvx[i] for i in range(6)]
        oe_mat["+vx"] = kepler_dynamics.sv_to_oe(state_vector=state_mat["+vx"], mass=self.conversion_mass)
        solution_mat["+vx"] = oe_mat["+vx"][0:2] + [oe_mat["+vx"][3] + oe_mat["+vx"][4] + oe_mat["+vx"][5]]

        dvy = [0, 0, 0, 0, dv, 0]
        state_mat["+vy"] = [state[i] + dvy[i] for i in range(6)]
        oe_mat["+vy"] = kepler_dynamics.sv_to_oe(state_vector=state_mat["+vy"], mass=self.conversion_mass)
        solution_mat["+vy"] = oe_mat["+vy"][0:2] + [oe_mat["+vy"][3] + oe_mat["+vy"][4] + oe_mat["+vy"][5]]

        dvz = [0, 0, 0, 0, 0, dv]
        state_mat["+vz"] = [state[i] + dvz[i] for i in range(6)]
        oe_mat["+vz"] = kepler_dynamics.sv_to_oe(state_vector=state_mat["+vz"], mass=self.conversion_mass)
        solution_mat["+vz"] = oe_mat["+vz"][0:2] + [oe_mat["+vz"][3] + oe_mat["+vz"][4] + oe_mat["+vz"][5]]

        # Filling in the empty parameters in the taret orbital element list
        target_oe_list = [0 for _ in range(len(oe_name_list))]
        for i, oe in enumerate(oe_name_list):
            if oe in self.target_oe.keys():
                target_oe_list[i] = self.target_oe[oe] - oe_mat["center"][i]
            else:
                target_oe_list[i] = 0

        target_oe_list = np.array(target_oe_list)

        jacobian = [[0, 0, 0] for _ in range(len(oe_name_list))]

        for i in range(len(oe_name_list)):
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

    def solar_sail_local_try_1(self, state, force_model, system_time):

        if self.target_oe == {}:
            raise ValueError("No target orbital parameter set defined!")

        oe_name_list = ["SMA", "ECC", "INC", "RAAN", "APERI", "TAEPO"]

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

        # Filling in the empty parameters in the taret orbital element list
        target_oe_list = [0 for _ in range(6)]
        for i, oe in enumerate(oe_name_list):
            if oe in self.target_oe.keys():
                target_oe_list[i] = (self.target_oe[oe] - oe_mat["center"][i]) / self.target_oe[oe]
            else:
                target_oe_list[i] = 0

        target_oe_list = np.array(target_oe_list)

        # Jacobian J
        jacobian = [[0, 0, 0] for _ in range(6)]

        for i in range(6):
            for j, delta_state in enumerate(["+vx", "+vy", "+vz"]):
                # We are now using the difference between target and current orbital elements for the Jacobian
                jacobian[i][j] = (oe_mat[delta_state][i] - oe_mat["center"][i]) / dv

        J = np.array(jacobian)

        J_T_p = np.linalg.pinv(J)

        # C matrix: Control matrix, mapping changes in sail attitude to changes in acceleration
        delta_ang = 0.001

        control_mat = dict()
        acc_mat = dict()

        control_mat["center"] = force_model.solar_pressure.sail_control
        acc_mat["center"] = force_model.solar_pressure.solar_acceleration(state=state[0:3])

        force_model.solar_pressure.sail_control[0] += delta_ang
        control_mat["+dt"] = force_model.solar_pressure.sail_control
        acc_mat["+dt"] = force_model.solar_pressure.solar_acceleration(state=state[0:3])

        force_model.solar_pressure.sail_control[0] -= delta_ang
        force_model.solar_pressure.sail_control[1] += delta_ang

        control_mat["+dc"] = force_model.solar_pressure.sail_control
        acc_mat["+dc"] = force_model.solar_pressure.solar_acceleration(state=state[0:3])

        force_model.solar_pressure.sail_control[1] -= delta_ang

        # Getting the control matrix C

        C = [[0, 0] for _ in range(3)]

        for i in range(3):
            for j, delta_control in enumerate(["+dt", "+dc"]):
                C[i][j] = (acc_mat[delta_control][i] - acc_mat["center"][i]) / delta_ang

        C_p = np.linalg.pinv(np.array(C))  # Pseudo_inverse of control matrix

        # Evaluating everything

        du = C_p.dot(J_T_p.dot(target_oe_list)) / self.time_constant
        du = du * min([0.01, np.linalg.norm(du)]) / np.linalg.norm(du)

        force_model.solar_pressure.sail_control = list(np.array(force_model.solar_pressure.sail_control) + du)

        # Fixing the sail control magnitude
        for i, ang in enumerate(force_model.solar_pressure.sail_control):
            ang = ang - 2 * math.pi * math.floor(ang / (2 * math.pi))
            force_model.solar_pressure.sail_control[i] = ang

        self.current_control = force_model.solar_pressure.solar_acceleration(state=state[0:3])
        return force_model.solar_pressure.sail_control

    def solar_sail_local(self, state, force_model, system_time):

        if self.target_oe == {}:
            raise ValueError("No target orbital parameter set defined!")

        oe_name_list = ["SMA", "ECC", "INC", "RAAN", "APERI", "TAEPO"]

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

        # Filling in the empty parameters in the taret orbital element list
        target_oe_delta_list = [0 for _ in range(6)]
        for i, oe in enumerate(oe_name_list):
            if oe in self.target_oe.keys():
                target_oe_delta_list[i] = (self.target_oe[oe] - oe_mat["center"][i])
            else:
                target_oe_delta_list[i] = 0

        target_oe_delta_list = np.array(target_oe_delta_list)

        # Jacobian J
        jacobian = [[0, 0, 0] for _ in range(6)]

        for i in range(6):
            for j, delta_state in enumerate(["+vx", "+vy", "+vz"]):
                # We are now using the difference between target and current orbital elements for the Jacobian
                jacobian[i][j] = (oe_mat[delta_state][i] - oe_mat["center"][i]) / dv

        J = np.array(jacobian)

        J_T_p = np.linalg.pinv(J)

        # C matrix: Control matrix, mapping changes in sail attitude to changes in acceleration
        delta_ang = 0.001

        control_mat = dict()
        acc_mat = dict()

        control_mat["center"] = force_model.solar_pressure.sail_control
        acc_mat["center"] = force_model.solar_pressure.solar_acceleration(state=state[0:3])

        force_model.solar_pressure.sail_control[0] += delta_ang
        control_mat["+dt"] = force_model.solar_pressure.sail_control
        acc_mat["+dt"] = force_model.solar_pressure.solar_acceleration(state=state[0:3])

        force_model.solar_pressure.sail_control[0] -= delta_ang
        force_model.solar_pressure.sail_control[1] += delta_ang

        control_mat["+dc"] = force_model.solar_pressure.sail_control
        acc_mat["+dc"] = force_model.solar_pressure.solar_acceleration(state=state[0:3])

        force_model.solar_pressure.sail_control[1] -= delta_ang

        # Getting the control matrix C

        C = [[0, 0] for _ in range(3)]

        for i in range(3):
            for j, delta_control in enumerate(["+dt", "+dc"]):
                C[i][j] = (acc_mat[delta_control][i] - acc_mat["center"][i]) / delta_ang

        C_p = np.linalg.pinv(np.array(C))  # Pseudo_inverse of control matrix

        # Evaluating everything
        if (system_time - self.prev_time) == 0:
            force_model.solar_pressure.sail_control = np.array([0, 0])
            # self.scaling_acc = np.linalg.norm(force_model.solar_pressure.solar_acceleration(state=state[0:3]))
            self.current_control = [0, 0, 0]
            self.prev_acc = self.current_control
            self.prev_time = system_time
        else:
            new_acc = J_T_p.dot(target_oe_delta_list) * (system_time - self.prev_time) ** (-1)
            new_acc = new_acc * np.linalg.norm(self.prev_acc) / np.linalg.norm(new_acc)
            delta_ddx = new_acc - self.prev_acc
            du = C_p.dot(delta_ddx)

            # Limit change in orientation within a given time step
            if np.linalg.norm(du) > 0.1:
                du = du * 0.1 / np.linalg.norm(du)
            force_model.solar_pressure.sail_control = force_model.solar_pressure.sail_control + du

            self.current_control = force_model.solar_pressure.solar_acceleration(state=state[0:3])
            self.prev_acc = self.current_control
            self.prev_time = system_time

        """du = C_p.dot(J_T_p.dot(target_oe_delta_list)) / self.time_constant
        du = du * min([0.01, np.linalg.norm(du)]) / np.linalg.norm(du)

        force_model.solar_pressure.sail_control = list(np.array(force_model.solar_pressure.sail_control) + du)"""

        # Fixing the sail control magnitude
        for i, ang in enumerate(force_model.solar_pressure.sail_control):
            ang = ang - 2 * math.pi * math.floor(ang / (2 * math.pi))
            force_model.solar_pressure.sail_control[i] = ang

        return force_model.solar_pressure.sail_control


    def guidance(self, state, time, force_model):

        arc_sun = (time / (24*3600*365)) * 2 * math.pi
        pos_sun = np.array([math.cos(arc_sun), math.sin(arc_sun), 0]) * 149000000000
        force_model.solar_pressure.radiation_location = pos_sun

        self.target_oe = {"SMA": 100000000000}

        target_vel_change = np.array(self.target_orbit(state=state))
        target_vel_change_aux = np.array(state[3:6])

        target_vel_change = target_vel_change / np.linalg.norm(target_vel_change) * np.linalg.norm(target_vel_change_aux)
        target_vel_change_aux = target_vel_change_aux / np.linalg.norm(target_vel_change_aux)

        sail_control, vel_angle = direct_control_inversion(vel_change=target_vel_change, state=state, force_model=force_model)

        force_model.solar_pressure.sail_control = sail_control
        self.current_control = force_model.solar_pressure.solar_acceleration(state=state[0:3])
        if not self.control_command_track:
            self.control_command_track = {"Tilt": [], "Clock": []}
        if not self.vel_angle_track:
            self.vel_angle_track = {"Target velocity tilt": [], "Target velocity clock": []}
        self.control_command_track["Tilt"].append(sail_control[0])
        self.control_command_track["Clock"].append(sail_control[1])

        self.vel_angle_track["Target velocity tilt"].append(vel_angle[0])
        self.vel_angle_track["Target velocity clock"].append(vel_angle[1])
        return self.current_control


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    d_1 = np.array([1, 0, 0])
    d_2 = np.array([0, -1, 0])
    d_3 = np.array([0, 0, 1])


    alpha_v_array= np.linspace(-2*math.pi, 2*math.pi, 1000)
    alpha = [math.atan((-3 + (9 + 8 * math.tan(alpha_v) ** 2) ** 0.5) / (4 * math.tan(alpha_v))) for alpha_v in alpha_v_array]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alpha_v_array, np.array(alpha))
    ax.set_ylabel("Alpha control")
    ax.set_xlabel("Alpha target")
    plt.show()

