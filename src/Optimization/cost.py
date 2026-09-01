import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from numpy.ma.core import argmin

import src.spacecraft.sc
from scenarios.presentation import integrator_comparison
from src.feasibility.valid_set import feasibility_setup
from src.system_dynamics import sd_1, SRP, atmo, eclipse, sd_rotating, R4BP_inertial
from src.analysis import plotting_functions
from src.spacecraft import swarm_1
from src.astrodynamic_functions import kepler_dynamics
from src.feasibility import valid_set
from src.guidance import steering_laws, events
from src.globals import Constants
from miscellaneous import writer_tools
import math
import pickle
from src.Optimization import Interface
G, MY, KS_TOLERANCE, GRAV_CONST, EARTH_RADIUS, OBLIQUITY = src.globals.Constants.get_globals()

from src.system_dynamics.atmo import EARTH_RADIUS

normalization_factors = Constants.get_normalization_factors()

def write_iteration(dof: list, cost_static: float, cost_integral: float, cost_total: float, intercept_params: list, initial_params: list, time: float):
    df_old = pd.read_csv("src/Optimization/iterations.csv")
    keys = df_old.keys()

    iter_vec = dof[0:-1] + [cost_static, cost_integral, cost_total, time] + [dof[-1]] + intercept_params + initial_params
    dict_new = dict()
    for i, key in enumerate(keys):
        dict_new[key] = [iter_vec[i]]
    df_new = pd.DataFrame.from_dict(data=dict_new)
    df_new.to_csv("src/Optimization/iterations.csv", mode='a', index=False, header=False)
    pass

def get_initial_states(intercept_parameters, itf_data):

    initial_states = ["r_init", "INC_init", "solar_phasing", "propagation_cutoff_SMA"]
    ret = dict()

    for state in initial_states:
        itp = Interface.Interface(discrete_interface=itf_data, interface_space=["SMA", "ECC", "INC", state], target_dimension=state)
        itp.target_dimension = state
        ret[state] = itp.interface_interpolation(eval_points=[intercept_parameters], fill_val=123456789)
        if ret[state] == 123456789:
            ret[state] = None
    return ret



def cost_function(dof_vec, plot_solution=False):
    # Notes for Lagrange setup:
    mu_star = 236530592967627.8

    # target_bounds = Interface.Interface(discrete_interface=pd.read_csv("./scenarios/Legacy/Interface_sigma_04.csv"), interface_space=["SMA", "ECC", "INC"]).get_convex_rectangle()

    SMA_track = list(np.array(dof_vec[0:10]) * normalization_factors["SMA"])
    ECC_track = list(np.array(dof_vec[10:20]) * normalization_factors["ECC"])
    INC_track = list(np.array(dof_vec[20:30]) * normalization_factors["INC"])
    t_offset = dof_vec[-1] * normalization_factors["time_offset"]
    # t_prop = dof_vec[15] * normalization_factors["t_prop"]

    t_start = normalization_factors["time_offset"]
    t_end = -30 * 24 * 3600
    integration_points = list(np.linspace(t_start, t_end, round(5000 * (t_start - t_end) / (-t_end))))
    earth_mass = 5.9722e24
    solar_mass = 1.989 * 10 ** 30

    # L1_state_vector = [306770476.4024763, 316770.58199267735, 0.0, -0.8958042029688121, 895.8039043673915, 81.62614187080287]

    itf_data = pd.read_csv("./scenarios/Legacy/Interface_sigma_04_full.csv")

    itf = Interface.Interface(discrete_interface=itf_data, interface_space=["SMA", "ECC", "INC", "t_s"], target_dimension="t_s")

    force_model = R4BP_inertial.R4BP_force_model(path="./data/R4BP_no_sync_circular.xlsx")
    force_model.define_central_attractor(mass=earth_mass, position=[0, 0, 0])
    force_model.central_attractor_gravity_law = src.astrodynamic_functions.kepler_dynamics.J_X_acceleration_tilted_to_ecliptic

    guidance_law = steering_laws.LocalOptimal()

    guidance_law.setup_continuouos_targeting(collocation_points={"SMA": SMA_track,
                                                                 "ECC": ECC_track,
                                                                 "INC": INC_track,
                                                                 "t": np.linspace(t_end, 0, len(SMA_track))})

    guidance_law.conversion_mass = earth_mass
    guidance_law.guidance_function = guidance_law.guidance_3_optic
    guidance_law.integration_direction = -1
    force_model.guidance = guidance_law

    Eclipse_interface = src.system_dynamics.eclipse.eclipse_model(eclipse_bodies={"central_attractor": 6378000}, force_model=force_model)
    force_model.guidance.eclipse_model = Eclipse_interface

    srp_model = SRP.Solar_pressure(sail_model="ACS3", central_attractor_mass=solar_mass, sigma=0.04)
    srp_model.radiation_location = [149000000000, 0, 0]
    srp_model.sail_control = [0, 0]

    force_model.solar_pressure = srp_model

    manifolds = [[-6197696.3949212525, -6197696.3949212525, 1],
                 [2638614.7315163864, 2638614.7315163864, 1],
                 [-753362.9238320779, -753362.9238320779, 1],
                 [-3074.0790258669726, -3074.0790258669726, 1],
                 [-6384.415594809771, -6384.415594809771, 1],
                 [2928.463010243008, 2928.463010243008, 1],
                 [0, 0, 1]]


    sw_1 = swarm_1.particle_swarm(manifolds, force_model)
    sw_1.do_integration = False
    sw_1.integration_points = integration_points
    sw_1.square_swarm('generic')
    sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=False, cores=11)
    # sw_1.get_swarm_body_distances(["Moon"])

    for i, sc in enumerate(sw_1.list_of_spacecraft):
        L1_state_vector = [318497940.45684016, 8.821101899835671, 0.8037832293497849, -2.3969568788805396e-05, 858.0975888840403, 78.19028267923386]
        L1_params = kepler_dynamics.sv_to_oe(state_vector=L1_state_vector, mass=mu_star / GRAV_CONST)
        L1_state_vector = kepler_dynamics.oe_to_sv(a=L1_params[0], e=L1_params[1], i=L1_params[2], RAAN=L1_params[3], APERI=L1_params[4], ny_0=L1_params[5], t=t_offset,
                                                   mass=mu_star / GRAV_CONST)
        sc.init_state_vector = L1_state_vector
        sc.time_interval = [t_offset, t_end + t_offset]
        sc.event_cutoff_val = itf
        # sc.display_name = str(round(float(RAAN_list[0]) * 180 / math.pi))

    sw_1.do_integration = True

    sw_1.create_and_integrate_swarm(rtol=1e-5, parproc=False, cores=11)

    if plot_solution:
        plots = plotting_functions.graph_output(list_of_spacecraft=[],
                                                list_of_resampled_spacecraft=[],
                                                list_of_special_spacecraft=sw_1.list_of_spacecraft,
                                                force_model=force_model,
                                                axis_visibility=False,
                                                animated=True)

        plots.trajectory_xyz()
        plots.parameters_plot(plot_reference_trajectory=True)
        plots.plot_steering_acceleration()
        plots.plot_control()
        plots.plot_drag_acceleration()
        plots.plot_target_velocity_angles()
        plots.magnitude_plot()
        plots.plot_drag_acceleration()
        # plots.body_distances_plot(body_list=["Moon", "Earth"])
        plots.C3_plot()
        plots.moving_map_plot(k_modulo=10, match_tail_color=True, override_limits={"x": [-500000000, 500000000], "y": [-500000000, 500000000], "z": [-500000000, 500000000]})
        # plots.moving_map_plot(match_tail_color=False)
        plt.show()
        plt.waitforbuttonpress(10000000000)

    for i, sc in enumerate(sw_1.list_of_spacecraft):
        # Find the point of the trajectory with the lowest total cost, also after propagating past the interface domain
        samples = [[sc.orbital_parameters_track[index][f] for index in range(3)] for f in range(len(integration_points))]
        t_interface = itf.interface_interpolation(eval_points=samples, fill_val=10 ** 10)
        t_integral = [integration_points[f] for f in range(len(integration_points))]
        t_total = np.array(t_interface) - np.array(t_integral)
        min_idx = argmin(t_total)

        # Get the initial point of the trajectory in LEO:
        ret = get_initial_states(intercept_parameters=[sc.orbital_parameters_track[index][min_idx] for index in range(3)], itf_data=itf_data)
        print("Minimum cost: ", round(min(t_total) * 10**(-7), 2), " at integration point ", min_idx)
        print("SMA: ", dof_vec[0:10])
        print("ECC: ", dof_vec[10:20])
        print("INC: ", dof_vec[20:30])
        print("Intercept parameters: ", [sc.orbital_parameters_track[index][min_idx] for index in range(6)])
        print("Initial parameters:")
        ret_list = []
        for key in ret.keys():
            print(key, ": ", ret[key])
            ret_list.append(ret[key])

        # write_iteration(dof=list(dof_vec), cost_static=t_interface[min_idx], cost_integral=t_integral[min_idx], cost_total=t_total[min_idx], intercept_params=[sc.orbital_parameters_track[index][min_idx] for index in range(3)], initial_params=ret_list, time=integration_points[argmin(t_total)])

        return min(t_total) * 10**(-7)
    pass

if __name__ == "__main__":
    SMA = [0.44009458, 0.68999098, 0.11857956, 0.14648585, 0.94829894, 0.66602284, 0.49277852, 0.53307753, 0.18328096, 0.62241777]
    ECC = [0.22654902, 0.56240003, 0.37091885, 0.68636426, 0.82615283, 0.3465491, 0.45132474, 0.70372842, 0.5082826, 0.25131128]
    INC = [0.99422308, 0.36819597, 0.7195283,  0.92318458, 0.953914, 0.62319311, 0.77797232, 0.64233334, 0.36426403, 1.22285744]
    dof = SMA + ECC + INC + [-58255.56359642568]
    tof = cost_function(dof)
    print(tof)