import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from src.Optimization import Interface

mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'

output_keys = ["SMA", "ECC", "INC", "RAAN", "APERI", "t_s"]
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


surface_plot_domain = ["solar_phasing", "INC_init"]  # TODO this needds to be two dimensions of the terminal space
sampling = 10

data_raw = pd.read_csv("../scenarios/Legacy/r_init__propagation_cutoff_SMA.csv")
data_raw = data_raw.drop('empty', axis=1)

# base_domain = get_data_dimensions(data_raw)
normalized_data = normalize_target_domain(data_raw)
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
params = []
for k in output_keys:
    if k != "t_s":
        params.append(k)




X, Y = np.meshgrid(np.linspace(bounds[surface_plot_domain[0]][0], bounds[surface_plot_domain[0]][1], sampling),
                   np.linspace(bounds[surface_plot_domain[1]][0], bounds[surface_plot_domain[1]][1], sampling))



plt.show()

