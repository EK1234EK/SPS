from scipy.optimize import direct
from scipy.stats import alpha

import src.system_dynamics.SRP
from src.astrodynamic_functions import kepler_dynamics
import numpy as np
import math
from src.system_dynamics import SRP

from src.astrodynamic_functions.kepler_dynamics import GRAV_CONST, EARTH_RADIUS


def kill_integrator_eccentricity(time, state):
    return kepler_dynamics.sv_to_oe(state_vector=state, mass=5.97e24)[1] - 0.99


def kill_integrator_SMA(time, state):
    return kepler_dynamics.sv_to_oe(state_vector=state, mass=5.97e24)[0] - 10 ** 10


def kill_integrator_C3(time, state):
    SMA = kepler_dynamics.sv_to_oe(state_vector=state, mass=5.97e24)[0]
    C3 = -5.97e24 * GRAV_CONST / SMA + 100000
    return C3


def kill_integrator_altitude(time, state):
    radius = np.linalg.norm(np.array(state[0:3]))
    return radius - EARTH_RADIUS - 100000


def vector_from_angle(alpha, gamma, d_1, d_2, d_3):
    return math.cos(alpha) * d_1 + math.sin(alpha) * math.sin(gamma) * d_2 + math.sin(alpha) * math.cos(gamma) * d_3


def angle_from_vector(v, d_1, d_2, d_3):
    alpha = math.acos((v.dot(d_1)) / (np.linalg.norm(v) * np.linalg.norm(d_1)))

    # v_d1 = (d_1 / (np.linalg.norm(d_1)) ** 2) * v.dot(d_1)  # Projection onto d_1
    # v_plane = v - v_d1  # Component in d_2 - d_3 plane
    # gamma = math.acos((d_3.dot(v_plane)) / (np.linalg.norm(d_3) * np.linalg.norm(v_plane)))

    v_d2 = ((d_2 / (np.linalg.norm(d_2)) ** 2) * v.dot(d_2)).dot(d_2 / (np.linalg.norm(d_2)))  # Projection onto d_2
    v_d3 = ((d_3 / (np.linalg.norm(d_3)) ** 2) * v.dot(d_3)).dot(d_3 / (np.linalg.norm(d_3)))  # Projection onto d_3
    gamma = math.atan2(v_d2, v_d3)
    if gamma < 0: gamma += 2 * math.pi

    return alpha, gamma


def direct_control_inversion(vel_change, state, force_model):
    # Attention only use with ideal sail!
    d_1, d_2, d_3, n = src.system_dynamics.SRP.sail_attitude([0, 0],
                                                             radiation_location=force_model.solar_pressure.radiation_location,
                                                             state=state[0:3])
    alpha_v, gamma_v = angle_from_vector(vel_change, d_1, d_2, d_3)

    if np.dot(n, vel_change) < 0:
        alpha = 0.5 * math.pi
        gamma = gamma_v
        return [alpha, gamma], [alpha_v, gamma_v], n
    gamma = gamma_v
    alpha = math.atan((-3 + (9 + 8 * math.tan(alpha_v) ** 2) ** 0.5) / (4 * math.tan(alpha_v)))
    if alpha < 0:
        alpha *= -1

    return [alpha, gamma], [alpha_v, gamma_v], n


def get_jacobian(oe_mat, dv):
    jacobian = [[0, 0, 0] for _ in range(6)]

    for i in range(6):
        for j, delta_state in enumerate(["+vx", "+vy", "+vz"]):
            # We are now using the difference between target and current orbital elements for the Jacobian
            if oe_mat["center"][i] == 0:
                jacobian[i][j] = (oe_mat[delta_state][i] - oe_mat["center"][i]) / dv
            else:
                jacobian[i][j] = (oe_mat[delta_state][i] - oe_mat["center"][i]) / (dv * oe_mat["center"][i])

    J = np.array(jacobian)
    return J


class LocalOptimal:
    def __init__(self):
        self.jacobian = [[]]
        self.target = ""
        self.conversion_mass = -1

        self.current_control = []

        self.oe_direction = []
        self.target_oe = {"True time": []}

        # Both need to be initialized before first integration step!
        self.control_command_track = dict()  # Write whatever you want into this at every interation
        self.vel_angle_track = dict()  # Same here

        self.terminator = None  # What function to use for termination of integration

        self.guidance_function = None  # What function is read by the force model to actually return the acceleration

        self.current_n = None  # Sail normal at the current time step

    def target_orbit_pinv_jacobian(self, state):

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

        J = get_jacobian(oe_mat=oe_mat, dv=dv)
        try:
            J_T_p = np.linalg.pinv(J)
        except:
            self.current_control = [0, 0, 0]
            return None

        # Filling in the empty parameters in the taret orbital element list
        target_oe_list = [0 for _ in range(6)]
        for i, oe in enumerate(oe_name_list):
            if oe in self.target_oe.keys():
                target_oe_list[i] = (self.target_oe[oe] - oe_mat["center"][i]) / oe_mat["center"][i]
            else:
                target_oe_list[i] = 0

        target_oe_list = np.array(target_oe_list)

        dr = J_T_p.dot(target_oe_list)
        mag = (dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2) ** 0.5
        if mag == 0:
            self.current_control = [0, 0, 0]
            return None

        direction = dr / mag
        direction = direction * acc_mag
        return direction

    def target_orbit_gradient(self, state, time):

        if self.target_oe == {}:
            raise ValueError("No target orbital parameter set defined!")

        oe_name_list = ["SMA", "ECC", "INC", "RAAN", "APERI", "TAEPO"]

        dv = 0.1  # Incement, by which to estimate the derivativs in the Jacobian

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

        J = get_jacobian(oe_mat=oe_mat, dv=dv)

        J_T = J.transpose()

        # Weights matrix:
        weights = np.zeros((6, 1))
        for i, oe in enumerate(oe_name_list):
            if oe in self.target_oe.keys():
                weights[i] = (self.target_oe[oe] - oe_mat["center"][i]) / oe_mat["center"][i]
            else:
                weights[i] = 0

        grad = np.transpose(np.dot(J_T, weights))[0]
        if time == 429345.29784990783:
            pass


        m = np.linalg.norm(grad)
        if m == 0:
            return np.array([0, 0, 0])

        return grad / m

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

    def guidance_1(self, state, time, force_model):

        arc_sun = (time / (24 * 3600 * 365)) * 2 * math.pi
        pos_sun = np.array([math.cos(arc_sun), math.sin(arc_sun), 0]) * 149000000000
        force_model.solar_pressure.radiation_location = pos_sun

        self.target_oe = {"SMA": 100000000000}

        target_vel_change = np.array(self.target_orbit_pinv_jacobian(state=state))

        target_vel_change = target_vel_change / np.linalg.norm(target_vel_change)

        sail_control, vel_angle, n = direct_control_inversion(vel_change=target_vel_change, state=state,
                                                              force_model=force_model)
        self.current_n = n

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

    def guidance_2(self, state, time, force_model):

        arc_sun = (time / (24 * 3600 * 365)) * 2 * math.pi
        pos_sun = np.array([math.cos(arc_sun), math.sin(arc_sun), 0]) * 149000000000
        force_model.solar_pressure.radiation_location = pos_sun

        self.target_oe = {"SMA": 100000000000}

        target_vel_change_aux = np.array(state[3:6])

        target_vel_change_aux = target_vel_change_aux / np.linalg.norm(target_vel_change_aux)

        sail_control, vel_angle, n = direct_control_inversion(vel_change=target_vel_change_aux, state=state,
                                                              force_model=force_model)
        self.current_n = n

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

    def guidance_3(self, state, time, force_model):
        arc_sun = (time / (24 * 3600 * 365)) * 2 * math.pi
        pos_sun = np.array([math.cos(arc_sun), math.sin(arc_sun), 0]) * 149000000000
        force_model.solar_pressure.radiation_location = pos_sun

        self.target_oe = {"SMA": 100000000000}

        target_vel_change = self.target_orbit_gradient(state=state)
        sail_control, vel_angle, n = direct_control_inversion(vel_change=target_vel_change, state=state,
                                                              force_model=force_model)
        self.current_n = n

        force_model.solar_pressure.sail_control = sail_control
        self.current_control = force_model.solar_pressure.solar_acceleration(state=state[0:3])
        if np.nan in self.current_control:
            pass
        if not self.control_command_track:
            self.control_command_track = {"Tilt": [], "Clock": []}
        if not self.vel_angle_track:
            self.vel_angle_track = {"Target velocity tilt": [], "Target velocity clock": []}
        self.control_command_track["Tilt"].append(sail_control[0])
        self.control_command_track["Clock"].append(sail_control[1])

        self.vel_angle_track["Target velocity tilt"].append(vel_angle[0])
        self.vel_angle_track["Target velocity clock"].append(vel_angle[1])
        return self.current_control

    def guidance_atmo(self, state, time, force_model):
        arc_sun = (time / (24 * 3600 * 365)) * 2 * math.pi
        pos_sun = np.array([math.cos(arc_sun), math.sin(arc_sun), 0]) * 149000000000
        force_model.solar_pressure.radiation_location = pos_sun

        self.target_oe = {"SMA": 100000000000}

        target_vel_change = self.target_orbit_gradient(state=state, time=time)
        if target_vel_change is None or np.nan in target_vel_change:
            # Some error during the control algorithm. It returns None
            vel_angle = [0, 0, 0]
            atmo_acc = force_model.drag_model.get_aero_acc(state=np.array(state), n=self.current_n, sigma=force_model.solar_pressure.sail_parameters["sigma"])
            self.current_n = np.cross(atmo_acc, force_model.solar_pressure.radiation_location - state[0:3])
            self.current_n = self.current_n / (np.linalg.norm(self.current_n))

            d_1_mod, d_2_mod, d_3_mod, _ = src.system_dynamics.SRP.sail_attitude([0, 0],
                                                                                 radiation_location=force_model.solar_pressure.radiation_location,
                                                                                 state=state[0:3])

            # Determine the sail control such that the sail normal actually has the correct orientation
            sail_control = angle_from_vector(v=self.current_n, d_1=d_1_mod, d_2=d_2_mod, d_3=d_3_mod)
            force_model.solar_pressure.sail_control = sail_control
            self.current_control = np.array([0, 0, 0])  # force_model.solar_pressure.solar_acceleration(state=state[0:3])

        else:
            # The happy path, target_vel_change is not None
            sail_control, vel_angle, n = direct_control_inversion(vel_change=target_vel_change, state=state,
                                                                  force_model=force_model)
            self.current_n = n

            force_model.solar_pressure.sail_control = sail_control
            self.current_control = force_model.solar_pressure.solar_acceleration(state=state[0:3])

            # Checking against the atmospheric acceleration
            atmo_acc = force_model.drag_model.get_aero_acc(state=np.array(state), n=self.current_n, sigma=force_model.solar_pressure.sail_parameters["sigma"])
            if np.dot(self.current_control + atmo_acc,  target_vel_change) < -1e-15:  # For reasons of numeric stability
                # In case the atmosphere makes everything worse, orient the sail in such a way that the sail normal
                # is orthogonal to both the sun direction as well as the incident atmosphere
                self.current_n = np.cross(atmo_acc, force_model.solar_pressure.radiation_location - state[0:3])
                self.current_n = self.current_n / ( np.linalg.norm(self.current_n))

                d_1_mod, d_2_mod, d_3_mod, _ = src.system_dynamics.SRP.sail_attitude([0, 0],
                                                      radiation_location=force_model.solar_pressure.radiation_location,
                                                      state=state[0:3])

                # Determine the sail control such that the sail normal actually has the correct orientation
                force_model.solar_pressure.sail_control = angle_from_vector(v=self.current_n, d_1=d_1_mod, d_2=d_2_mod, d_3=d_3_mod)
                self.current_control = np.array([0, 0, 0])  # force_model.solar_pressure.solar_acceleration(state=state[0:3])


        if not self.control_command_track:
            self.control_command_track = {"Tilt": [], "Clock": []}
        if not self.vel_angle_track:
            self.vel_angle_track = {"Target velocity tilt": [], "Target velocity clock": []}
        self.control_command_track["Tilt"].append(sail_control[0])
        self.control_command_track["Clock"].append(sail_control[1])

        self.vel_angle_track["Target velocity tilt"].append(vel_angle[0])
        self.vel_angle_track["Target velocity clock"].append(vel_angle[1])
        return self.current_control

    def guidance_test(self, state, time, force_model):
        force_model.solar_pressure.radiation_location = [0, 0, 0]
        self.current_n = np.array(state[3:6]) / np.linalg.norm(np.array(state[3:6]))
        force_model.solar_pressure.sail_control = [0, 0]
        return np.array([0, 0, 0])

    def guidance(self, state, time, force_model):
        return self.guidance_function(state, time, force_model)


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt

    d_1 = np.array([1, 0, 0])
    d_2 = np.array([0, -1, 0])
    d_3 = np.array([0, 0, 1])

    alpha_v_array = np.linspace(-2 * math.pi, 2 * math.pi, 1000)
    alpha = [math.atan((-3 + (9 + 8 * math.tan(alpha_v) ** 2) ** 0.5) / (4 * math.tan(alpha_v))) for alpha_v in
             alpha_v_array]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alpha_v_array, np.array(alpha))
    ax.set_ylabel("Alpha control")
    ax.set_xlabel("Alpha target")
    plt.show()
