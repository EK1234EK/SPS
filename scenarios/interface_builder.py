import random
from ctypes.wintypes import SMALL_RECT

import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd

import src.spacecraft.sc
from src.system_dynamics import sd_1, SRP, atmo
from src.analysis import plotting_functions
from src.spacecraft import swarm_1
from src.astrodynamic_functions import kepler_dynamics
from src.feasibility import valid_set
from src.guidance import steering_laws, events
from miscellaneous import writer_tools
import math
import pickle

from src.system_dynamics.atmo import EARTH_RADIUS

def create_initial_states(manifolds: dict, ):
    n_s_vecs = 1
    for key in manifolds.keys():
        n_s_vecs *= manifolds[key][2]

    states = [np.array() for _ in range(n_s_vecs)]



def interface():
    t_start = 0
    t_end = 5000 * 24 * 3600
    integration_points = list(np.linspace(t_start, t_end, 10000))
    earth_mass = 5.9722e24
    solar_mass = 1.989 * 10 ** 30

    inp = input("Load pickle? (y)")
    if inp == "y":
        force_model = pickle.load(open('.p', 'rb'))
        sc_list = pickle.load(open('sv.p', 'rb'))
    else:
        print("Integrating all initial conditions")

    force_model = sd_1.inertial_force_model(path="./data/Moon.xlsx")
    force_model.define_central_attractor(mass=earth_mass, position=[0, 0, 0])
    force_model.central_attractor_gravity_law = src.astrodynamic_functions.kepler_dynamics.J_X_acceleration_equator_reference
    srp_model = SRP.Solar_pressure(sail_model="ACS3", central_attractor_mass=solar_mass, sigma=0.02)
    srp_model.radiation_location = [149000000000, 0, 0]
    srp_model.sail_control = [0, 0]
    force_model.solar_pressure = srp_model

    guidance_law = steering_laws.LocalOptimal()
    guidance_law.conversion_mass = earth_mass
    guidance_law.guidance_function = guidance_law.guidance_3_optic
    terminator = src.guidance.events.kill_integrator_SMA
    guidance_law.terminator = terminator
    force_model.guidance = guidance_law

    drag_model = atmo.Atmopshere()
    drag_model.static_drag = 0
    # force_model.drag_model = drag_model

    manifolds = [[-6197696.3949212525, -6197696.3949212525, 1],
                 [2638614.7315163864, 2638614.7315163864, 1],
                 [-753362.9238320779, -753362.9238320779, 1],
                 [-3074.0790258669726, -3074.0790258669726, 1],
                 [-6384.415594809771, -6384.415594809771, 22],
                 [2928.463010243008, 2928.463010243008, 1],
                 [0, 0, 1]]

    inc_list_base = np.linspace(3 * math.pi / 180, 28.5 * math.pi / 180, manifolds[4][2])
    init_dist_list_base = np.linspace(5000000, 5000000, 1)
    solar_phasing_list_base = np.linspace(0, 1.5 * math.pi, 20)
    integration_cutoff_base = np.linspace(100000000, 100000000, 1)

    init_result_df = pd.DataFrame()
    init_result_df["r_init"] = []
    init_result_df["INC_init"] = []
    init_result_df["solar_phasing"] = []
    init_result_df["propagation_cutoff_SMA"] = []
    init_result_df["SMA"] = []
    init_result_df["ECC"] = []
    init_result_df["INC"] = []
    init_result_df["t_s"] = []

    init_result_df.to_csv("interface.csv")

    for j_1 in range(len(init_dist_list_base)):
        for j_2 in range(len(solar_phasing_list_base)):
            for j_3 in range(len(integration_cutoff_base)):


                rand_inc_list = np.array([random.uniform(-0.5 * float(inc_list_base[0] - inc_list_base[1]),
                                                               0.5 * float(inc_list_base[0] - inc_list_base[1])) for i
                                                in
                                                range(len(inc_list_base))]) if len(
                    inc_list_base) > 1 else np.array([0])
                rand_init_dist_list = np.array([random.uniform(-0.5 * float(init_dist_list_base[0] - init_dist_list_base[1]),
                                          0.5 * float(init_dist_list_base[0] - init_dist_list_base[1])) for i in
                           range(len(init_dist_list_base))]) if len(init_dist_list_base) > 1 else np.array(
                    [0])
                rand_solar_phasing_list = np.array([
                    random.uniform(-0.5 * float(solar_phasing_list_base[0] - solar_phasing_list_base[1]),
                                   0.5 * float(solar_phasing_list_base[0] - solar_phasing_list_base[1])) for
                    i in range(len(solar_phasing_list_base))]) if len(
                    solar_phasing_list_base) > 1 else np.array([0])
                rand_integration_cutoff = np.array([
                    random.uniform(-0.5 * float(integration_cutoff_base[0] - integration_cutoff_base[1]),
                                   0.5 * float(integration_cutoff_base[0] - integration_cutoff_base[1])) for
                    i in range(len(integration_cutoff_base))]) if len(
                    integration_cutoff_base) > 1 else np.array([0])

                inc_list = inc_list_base + rand_inc_list
                init_dist_list = init_dist_list_base + rand_init_dist_list
                solar_phasing_list = solar_phasing_list_base + rand_solar_phasing_list
                integration_cutoff = integration_cutoff_base + rand_integration_cutoff

                force_model.guidance.initial_solar_phasing = solar_phasing_list[j_2]

                # Output arrays
                r_init = []
                init_INC = []
                solar_phasing = []
                propagation_cutoff_SMA = []

                SMA = []
                ECC = []
                INC = []
                t_s = []

                sw_1 = swarm_1.particle_swarm(manifolds, force_model)
                sw_1.do_integration = False
                sw_1.integration_points = integration_points
                sw_1.square_swarm('generic')
                sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)
                # sw_1.get_swarm_body_distances(["Moon"])

                for i, sc in enumerate(sw_1.list_of_spacecraft):
                    sc.event_cutoff_val = integration_cutoff[j_3]
                    sc.init_state_vector = kepler_dynamics.oe_to_sv(EARTH_RADIUS + init_dist_list[j_1], 0.001,
                                                                    inc_list[i], 3, 3, 3, 0, earth_mass)
                    # sc.display_name = str(round(float(inc_list[i]) * 180 /

                sw_1.do_integration = True

                sw_1.create_and_integrate_swarm(rtol=1e-5, parproc=True, cores=11)
                for i, sc in enumerate(sw_1.list_of_spacecraft):
                    print(sc.display_name, " Inclination: ", round(float(inc_list[i]), 3), " Terminal distance: ",
                          sc.slant_range_track[-1], end="")
                    try:
                        print(sc.event_time[0][0])
                    except:
                        print("No event trigger")
                        pass

                # Extracting the state data from the setup
                for i, sc in enumerate(sw_1.list_of_spacecraft):
                    # Determining the final time:
                    terminal_index = None
                    if sc.event_time:
                        for k in range(1, len(integration_points) - 1):
                            if integration_points[k - 1] <= sc.event_time[0][0] <= integration_points[k]:
                                terminal_index = k - 1

                    r_init.append(float(sc.orbital_parameters_track[0][0]))
                    init_INC.append(float(inc_list[i]))
                    solar_phasing.append(float(solar_phasing_list[j_2]))
                    propagation_cutoff_SMA.append(float(integration_cutoff[j_3]))

                    if terminal_index is not None:
                        SMA.append(float(sc.orbital_parameters_track[0][terminal_index]))
                        ECC.append(float(sc.orbital_parameters_track[1][terminal_index]))
                        INC.append(float(sc.orbital_parameters_track[2][terminal_index]))
                        t_s.append(float(sc.event_time[0][0]))
                    else:
                        SMA.append(None)
                        ECC.append(None)
                        INC.append(None)
                        t_s.append(None)

                out_df = pd.DataFrame()
                out_df["r_init"] = r_init
                out_df["INC_init"] = init_INC
                out_df["solar_phasing"] = solar_phasing
                out_df["propagation_cutoff_SMA"] = propagation_cutoff_SMA
                out_df["SMA"] = SMA
                out_df["ECC"] = ECC
                out_df["INC"] = INC
                out_df["t_s"] = t_s

                excel_data = pd.read_csv("interface.csv")
                excel_data = pd.concat([excel_data, out_df], ignore_index=True)
                excel_data.to_csv("interface.csv", index=False)

                print("Initial radius: ", r_init)
                print("Initial inclination: ", init_INC)
                print("Solar phasing: ", solar_phasing)
                print("Propagation cutoff SMA: ", propagation_cutoff_SMA)
                print("Temrinal SMA: ", SMA)
                print("Terminal ECC: ", ECC)
                print("Terminal INC: ", INC)
                print("Propagation time: ", t_s)

    inp = input("Save= (y / n)")
    if inp == "y":
        # Safe the stuff with pickle
        pickle.dump(sw_1.list_of_spacecraft, open('sv.p', 'wb'))
        pickle.dump(force_model, open('.p', 'wb'))

    input("Start plotting?")

    plots = plotting_functions.graph_output(list_of_spacecraft=[],
                                            list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=sw_1.list_of_spacecraft,
                                            force_model=force_model,
                                            axis_visibility=True,
                                            animated=False)

    plots.trajectory_xyz()
    plots.parameters_plot()
    plots.plot_steering_acceleration()
    plots.plot_control()
    plots.plot_target_velocity_angles()
    plots.magnitude_plot()
    plots.plot_drag_acceleration()
    plots.C3_plot()
    plots.moving_map_plot(k_modulo=10, match_tail_color=True)
    # plots.moving_map_plot(match_tail_color=False)
    plt.show()
    plt.waitforbuttonpress(10000000000)

if __name__ == "__main__":
    interface()