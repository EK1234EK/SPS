from cProfile import label

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import copy

from src.Optimization import Interface

mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'

output_keys = ["SMA", "ECC", "INC", "RAAN", "APERI", "t_s"]
# output_keys = ["SMA", "ECC", "INC", "t_s"]
input_keys = ["r_init", "INC_init", "solar_phasing", "propagation_cutoff_SMA"]

def get_data_dimensions(df):
    base_domain_keys = ["r_init", "INC_init", "solar_phasing", "propagation_cutoff_SMA"]

    base_domain_reorder = dict()
    for key in base_domain_keys:
        res = list(set(df[key].tolist()))
        res = set([round(res[i], 5) for i in range(len(res))])
        vals = np.linspace(min(res), max(res), len(res))
        base_domain_reorder[key] = vals
    return base_domain_reorder

def normalize_target_domain(df):
    normalized_df = pd.DataFrame()
    """output_keys = ["SMA", "ECC", "INC", "t_s"]
    input_keys = ["r_init", "INC_init", "solar_phasing", "propagation_cutoff_SMA"]"""
    for key in input_keys:
        normalized_df[key] = df[key]
    for key in output_keys:
        normalized_df[key] = list(np.array(df[key].tolist()) / max(df[key].tolist()))
    return normalized_df

def create_sample_points(params, sampling):
    import itertools

    sample_dict =dict()
    plot_dict = dict()
    comb_sample = itertools.product(params, params)

    all_par_plot_dict = dict()
    all_par_points = []

    all_par_disc = []
    for p in params:
        all_par_disc.append(list(np.linspace(bounds[p][0], bounds[p][1], sampling)))

    # Create the vectors that include all dimensions:
    var_par = itertools.product(*all_par_disc)

    for i, key in enumerate(params):
        all_par_plot_dict[key] = []

    for vp, par_set in enumerate(var_par):
        all_par_points.append(par_set)
        for i, key in enumerate(params):
            all_par_plot_dict[key].append(par_set[i])

# Generating 2d slices

    for cs in comb_sample:
        par_disc = []

        if cs[0] != cs[1]:
            dct = []
            for p in params:
                # all_par_disc.append(np.linspace(bounds[p][0], bounds[p][1], sampling))
                if p in cs:
                   dct.append(np.linspace(bounds[p][0], bounds[p][1], sampling))
                   par_disc.append(np.linspace(bounds[p][0], bounds[p][1], sampling))
                else:
                    dct.append([0.3 * (bounds[p][0] + bounds[p][1])])

            ep_iter = itertools.product(*dct)
            expaded_points = [e for e in ep_iter]
            sample_dict[cs[0] + cs[1]] = expaded_points

            # Get the 1d plotting arrays
            dim_arr_1 =[]
            dim_arr_2 = []
            for a_1 in par_disc[0]:
                for a_2 in par_disc[1]:
                    dim_arr_1.append(a_1)
                    dim_arr_2.append(a_2)
            plot_dict[cs[0] + cs[1]] = [dim_arr_1, dim_arr_2]


    return  sample_dict, plot_dict, all_par_points, all_par_plot_dict


surface_plot_domain = ["solar_phasing", "INC_init"]  # TODO this needds to be two dimensions of the terminal space
sampling = 10

data_raw = pd.read_csv("../scenarios/Legacy/Interface_sigma_04.csv")
data_raw = data_raw.drop('empty', axis=1)

# base_domain = get_data_dimensions(data_raw)
normalized_data = normalize_target_domain(data_raw)
normalized_data = data_raw
transform_data = ["INC_init", "solar_phasing"]

for k in transform_data:
    normalized_data[k] = list(np.array(normalized_data[k]) * 180 / math.pi)

# Get interpolated data:
connector = Interface.Interface(discrete_interface=normalized_data, interface_states=output_keys)
bounds = connector.get_convex_rectangle()

# Plot the shit

colors = {"INC": [1, 0, 0], "SMA": [1, 0, 1], "ECC": [0, 0, 1], "t_s": [0, 1, 0], "APERI": [1, 0.5, 0.5], "RAAN": [0, 1, 1]}

fig_1 = plt.figure()
fig_2= plt.figure()
fig_3 = plt.figure()
fig_4 = plt.figure()

axes = []

for i in range(len(input_keys)):
    axis = fig_1.add_subplot(math.ceil(len(input_keys)**0.5), math.ceil(len(input_keys)**0.5), i + 1)
    axis.set_title(input_keys[i])
    for j in range(len(output_keys)):
        axis.scatter(normalized_data[input_keys[i]].tolist(), normalized_data[output_keys[j]].tolist(), label=output_keys[j], color=colors[output_keys[j]], marker=".")
        # axis.set_xlabel(input_keys[i])
        axis.set_ylabel("Output")
    axes.append(axis)
axes[0].legend()

# 3D plots
for i in range(len(output_keys)):
    axis = fig_2.add_subplot(math.ceil(len(output_keys)**0.5), math.ceil(len(output_keys)**0.5), i + 1, projection="3d")
    axis.set_title(output_keys[i])
    axis.scatter(normalized_data[surface_plot_domain[0]].tolist(), normalized_data[surface_plot_domain[1]].tolist(), normalized_data[output_keys[i]].tolist(), label=output_keys[i], color=colors[output_keys[i]], marker=".")
    axis.set_xlabel("Solar phasing")
    axis.set_ylabel("INC_init")
    axis.set_zlabel(output_keys[i])

# Output parameter vs. output parameter plots
axes = []
iter = 1
for i_1, param_1 in enumerate(output_keys):
    for i_2, param_2 in enumerate(output_keys):
        ax = fig_3.add_subplot(len(output_keys), len(output_keys), iter)
        ax.scatter(normalized_data[param_1], normalized_data[param_2], label=param_1 + " - " + param_2, marker=".")
        ax.set_xlabel(param_1)
        ax.set_ylabel(param_2)
        axes.append(ax)
        iter += 1

# t_s vs. output parameter surface plots
# First, get the output parameters that are NOT constant:
domain_params = copy.deepcopy(output_keys)
domain_params.remove("t_s")
resampled_dict, plot_dict, all_par_points, all_par_plot_dict = create_sample_points(params=domain_params, sampling=20)


idx = 1
for d_1, param_1 in enumerate(domain_params):
    for d_2, param_2 in enumerate(domain_params):
        if param_1 != param_2 and d_2 < d_1:
            print(idx)
            # eval = connector.interface_interpolation(eval_points=resampled_dict[param_1 + param_2], fill_val=69)
            eval = connector.interface_interpolation(eval_points=all_par_points, fill_val=69)
            eval = [e if e != 69 else float(np.nan) for e in eval]
            ax = fig_4.add_subplot(len(domain_params), len(domain_params), idx, projection="3d")
            ax.scatter(normalized_data[param_1], normalized_data[param_2], normalized_data["t_s"], label="Ground truth", marker=".", alpha=1, color=[0, 0, 1])
            # ax.scatter(plot_dict[param_1+param_2][1], plot_dict[param_1+param_2][0], eval, label="Interpolated", marker=".", color=[1, 0, 0])
            ax.scatter(all_par_plot_dict[param_1], all_par_plot_dict[param_2], eval, label="Interpolated", marker=".", color=[1, 0, 0])
            ax.set_xlabel(param_1)
            ax.set_ylabel(param_2)
            # ax.view_init(0, 90, 0)
        else:
            ax = fig_4.add_subplot(len(domain_params), len(domain_params), idx, projection="3d")
            ax.scatter([], [], [])
        idx += 1
fig_4.legend()
plt.show()

