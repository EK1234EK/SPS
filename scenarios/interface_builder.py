import copy
import random
from ctypes.wintypes import SMALL_RECT

import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from PIL.ImageSequence import all_frames
from numba.core.unsafe.eh import exception_check

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

def randomize(arr: np.array):
    rand_list = np.array([random.uniform(0, 0.5 * float(arr[1] - arr[0]))] + [random.uniform(-0.5 * float(arr[1] - arr[0]),
    0.5 * float(arr[1] - arr[0])) for i in range(len(arr) - 2)] + [random.uniform(-0.5 * float(arr[1] - arr[0]), 0)]) if len(arr) > 1 else np.array([0])
    arr = arr + rand_list
    return arr

def create_initial_states(manifolds: dict):
    # Dict: {"Name", [lower, upper, samples, random]}
    n_s_vecs = 1
    keys = list(manifolds.keys())
    for key in manifolds.keys():
        n_s_vecs *= manifolds[key][2]

    states = []  # [np.zeros(len(keys)) for _ in range(n_s_vecs)]

    range_arr_list = [np.linspace(manifolds[key][0], manifolds[key][1], manifolds[key][2]) for key in keys]

    for i_1 in range(manifolds[keys[0]][2]):
        for i_2 in range(manifolds[keys[1]][2]):
            for i_3 in range(manifolds[keys[2]][2]):
                for i_4 in range(manifolds[keys[3]][2]):
                    idx = [i_1, i_2, i_3, i_4]
                    s_vec = np.zeros(len(keys))
                    for k in range(len(range_arr_list)):
                        if manifolds[keys[k]][3] == 1:
                            rand_arr = randomize(arr=copy.deepcopy(range_arr_list[k]))
                            s_vec[k] = rand_arr[idx[k]]
                        else:
                            s_vec[k] = range_arr_list[k][idx[k]]
                    states.append(s_vec)
    return np.array(states)


def interface():
    t_start = 0
    t_end = 5000 * 24 * 3600
    integration_points = list(np.linspace(t_start, t_end, 3000))
    earth_mass = 5.9722e24
    solar_mass = 1.989 * 10 ** 30

    inp = input("Load pickle? (y)")
    if inp == "y":
        force_model = pickle.load(open('.p', 'rb'))
        sc_list = pickle.load(open('sv.p', 'rb'))
    else:
        print("Integrating all initial conditions")

    force_model = sd_1.inertial_force_model(path="./../data/Moon.xlsx")
    force_model.define_central_attractor(mass=earth_mass, position=[0, 0, 0])
    force_model.central_attractor_gravity_law = src.astrodynamic_functions.kepler_dynamics.J_X_acceleration_tilted_to_ecliptic
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

    # Getting the states

    r_init__INC_init = {
        "r_init": [20000000, 20000000, 1, 0],
        "INC_init": [15*math.pi/180, 15 * math.pi / 180, 1, 0],
        "solar_phasing": [0, 1.5*math.pi, 25, 1],
        "propagation_cutoff_SMA": [50000000, 200000000, 25, 1],
    }

    r_init__solar_phasing = {
        "r_init": [20000000, 20000000, 1, 0],
        "INC_init": [5 * math.pi / 180, 18.5 * math.pi / 180, 25, 1],
        "solar_phasing": [0, 0, 1, 0],
        "propagation_cutoff_SMA": [50000000, 200000000, 25, 1],
    }

    r_init__propagation_cutoff_SMA = {
        "r_init": [20000000, 20000000, 1, 0],
        "INC_init": [5 * math.pi / 180, 18.5 * math.pi / 180, 25, 1],
        "solar_phasing": [0, 1.5*math.pi, 25, 1],
        "propagation_cutoff_SMA": [100000000, 100000000, 1, 0]
    }

    INC_init__solar_phasing = {
        "r_init": [EARTH_RADIUS + 500000, 20000000, 25, 1],
        "INC_init": [15 * math.pi / 180, 15 * math.pi / 180, 1, 0],
        "solar_phasing": [0, 0, 1, 0],
        "propagation_cutoff_SMA": [50000000, 200000000, 25, 1],
    }

    INC_init__propagation_cutoff_SMA = {
        "r_init": [EARTH_RADIUS + 500000, 20000000, 25, 1],
        "INC_init": [15 * math.pi / 180, 15 * math.pi / 180, 1, 0],
        "solar_phasing": [0, 1.5*math.pi, 25, 1],
        "propagation_cutoff_SMA": [100000000, 100000000, 1, 0],
    }

    solar_phasing__propagation_cutoff_SMA = {
        "r_init": [EARTH_RADIUS + 500000, 20000000, 25, 1],
        "INC_init": [5 * math.pi / 180, 18.5 * math.pi / 180, 25, 1],
        "solar_phasing": [0, 0, 1, 0],
        "propagation_cutoff_SMA": [100000000, 100000000, 1, 0],
    }

    all_grid_1 = {
        "r_init": [EARTH_RADIUS + 500000, 20000000, 7, 1],
        "INC_init": [5 * math.pi / 180, 18.5 * math.pi / 180, 7, 1],
        "solar_phasing": [0, 0.5*math.pi, 3, 1],
        "propagation_cutoff_SMA": [50000000, 200000000, 7, 1],
    }

    all_grid_2 = {
        "r_init": [EARTH_RADIUS + 500000, 20000000, 7, 1],
        "INC_init": [5 * math.pi / 180, 18.5 * math.pi / 180, 7, 1],
        "solar_phasing": [0.55 * math.pi, 1 * math.pi, 3, 1],
        "propagation_cutoff_SMA": [50000000, 200000000, 7, 1],
    }

    all_grid_3 = {
        "r_init": [EARTH_RADIUS + 500000, 20000000, 7, 1],
        "INC_init": [5 * math.pi / 180, 18.5 * math.pi / 180, 7, 1],
        "solar_phasing": [1.05 * math.pi, 1.5 * math.pi, 3, 1],
        "propagation_cutoff_SMA": [50000000, 200000000, 7, 1],
    }

    data_sets = {
        "r_init__INC_init": r_init__INC_init,
        "r_init__solar_phasing": r_init__solar_phasing,
        "r_init__propagation_cutoff_SMA": r_init__propagation_cutoff_SMA,
        "INC_init__solar_phasing": INC_init__solar_phasing,
        "INC_init__propagation_cutoff_SMA": INC_init__propagation_cutoff_SMA,
        "solar_phasing__propagation_cutoff_SMA": solar_phasing__propagation_cutoff_SMA,
    }

    data_sets = {
        "All_grid_1": all_grid_1,
        "All_grid_2": all_grid_2,
        "All_grid_3": all_grid_3
    }

    # =================================================== #
    # Generating a set of two fixed dimensions for each of the combinations of four degrees of freedom.
    # =================================================== #

    test_cases = data_sets.keys()

    for test_idx, case in enumerate(test_cases):
        try:
            manifolds_states = data_sets[case]

            states = create_initial_states(manifolds_states)

            manifolds = [[-6197696.3949212525, -6197696.3949212525, 1],
                         [2638614.7315163864, 2638614.7315163864, 1],
                         [-753362.9238320779, -753362.9238320779, 1],
                         [-3074.0790258669726, -3074.0790258669726, 1],
                         [-6384.415594809771, -6384.415594809771, len(states)],
                         [2928.463010243008, 2928.463010243008, 1],
                         [0, 0, 1]]

            result_df = dict()
            result_df["r_init"] = []
            result_df["INC_init"] = []
            result_df["solar_phasing"] = []
            result_df["propagation_cutoff_SMA"] = []
            result_df["SMA"] = []
            result_df["ECC"] = []
            result_df["INC"] = []
            result_df["RAAN"] = []
            result_df["APERI"] = []
            result_df["t_s"] = []

            # result_df.to_csv("interface.csv")

            sw_1 = swarm_1.particle_swarm(manifolds, force_model)
            sw_1.do_integration = False
            sw_1.integration_points = integration_points
            sw_1.square_swarm('generic')
            sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)

            for s, state in enumerate(states):
                # State = [r_init, INC_init, solar_phasing, propagation_cutoff_SMA]

                # Saving the initial domain
                result_df["r_init"].append(state[0])
                result_df["INC_init"].append(state[1])
                result_df["solar_phasing"].append(state[2])
                result_df["propagation_cutoff_SMA"].append(state[3])

                sc = sw_1.list_of_spacecraft[s]

                sc.force_model.guidance.initial_solar_phasing = state[2]

                sc.event_cutoff_val = state[3]
                sc.init_state_vector = kepler_dynamics.oe_to_sv(state[0], 0.001, state[1], 3, 3, 3, 0, earth_mass)

            sw_1.do_integration = True

            sw_1.create_and_integrate_swarm(rtol=1e-5, parproc=True, cores=11)

            # Extracting the state data from the setup
            for i, sc in enumerate(sw_1.list_of_spacecraft):
                # Determining the final time:
                terminal_index = None
                if sc.event_time:
                    for k in range(1, len(integration_points) - 1):
                        if integration_points[k - 1] <= sc.event_time[0][0] <= integration_points[k]:
                            terminal_index = k - 1

                print("Inclination: ", round(float(states[i][1]), 3), " Terminal distance: ", round(sc.slant_range_track[terminal_index]), end="")
                try:
                    print("  ", sc.event_time[0][0])
                except:
                    print("No event trigger")
                    pass


                if terminal_index is not None:
                    result_df["SMA"].append(float(sc.orbital_parameters_track[0][terminal_index]))
                    result_df["ECC"].append(float(sc.orbital_parameters_track[1][terminal_index]))
                    result_df["INC"].append(float(sc.orbital_parameters_track[2][terminal_index]))
                    result_df["RAAN"].append(float(sc.orbital_parameters_track[3][terminal_index]))
                    result_df["APERI"].append(float(sc.orbital_parameters_track[4][terminal_index]))
                    result_df["t_s"].append(float(sc.event_time[0][0]))
                else:
                    result_df["SMA"].append(None)
                    result_df["ECC"].append(None)
                    result_df["INC"].append(None)
                    result_df["RAAN"].append(None)
                    result_df["APERI"].append(None)
                    result_df["t_s"].append(None)

            result_df = pd.DataFrame.from_dict(result_df)
            try:
                excel_data = pd.read_csv(case + ".csv")
                excel_data = pd.concat([excel_data, result_df], ignore_index=True)
                excel_data.to_csv("interface.csv", index=False)
            except:
                result_df.to_csv(case + ".csv")

        except:
            print("No data saved for test case ", case)


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
    # plt.waitforbuttonpress(10000000000)

    inp = input("Save= (y / n)")
    if inp == "y":
        # Safe the stuff with pickle
        pickle.dump(sw_1.list_of_spacecraft, open('sv.p', 'wb'))
        pickle.dump(force_model, open('.p', 'wb'))

if __name__ == "__main__":
    interface()