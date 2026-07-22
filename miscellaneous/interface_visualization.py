import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'

def get_data_dimensions(df):
    keys = df.keys().tolist()
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
    output_keys = ["SMA", "ECC", "INC", "t_s"]
    input_keys = ["r_init", "INC_init", "solar_phasing", "propagation_cutoff_SMA"]
    for key in input_keys:
        normalized_df[key] = df[key]
    for key in output_keys:
        normalized_df[key] = list(np.array(df[key].tolist()) / max(df[key].tolist()))
    return normalized_df

data_raw = pd.read_csv("../interface.csv")
data_raw = data_raw.drop('Unnamed: 0', axis=1)

# base_domain = get_data_dimensions(data_raw)
normalized_data = normalize_target_domain(data_raw)
transform_data = ["INC_init", "solar_phasing"]

for k in transform_data:
    normalized_data[k] = list(np.array(normalized_data[k]) * 180 / math.pi)


# Plot the shit
output_keys = ["INC", "SMA", "ECC", "t_s"]
input_keys = ["r_init", "INC_init", "solar_phasing", "propagation_cutoff_SMA"]

colors = {"INC": [1, 0, 0], "SMA": [1, 0, 1], "ECC": [0, 0, 1], "t_s": [0, 1, 0]}

fig_1 = plt.figure()
fig_2= plt.figure()
axes = []
for i in range(len(input_keys)):
    axis = fig_1.add_subplot(2, 2, i + 1)
    axis.set_title(input_keys[i])
    for j in range(len(output_keys)):
        print(output_keys[j])
        axis.scatter(normalized_data[input_keys[i]].tolist(), normalized_data[output_keys[j]].tolist(), label=output_keys[j], color=colors[output_keys[j]])
        # axis.set_xlabel(input_keys[i])
        axis.set_ylabel("Output")
    axes.append(axis)
axes[0].legend()

# 3D plots
axes = []
for i in range(len(output_keys)):
    axis = fig_2.add_subplot(2, 2, i+1, projection="3d")
    axis.set_title(output_keys[i])
    axis.scatter(normalized_data["solar_phasing"].tolist(), normalized_data["INC_init"].tolist(), normalized_data[output_keys[i]].tolist(), label=output_keys[i], color=colors[output_keys[i]])
    axis.set_xlabel("Solar phasing")
    axis.set_ylabel("INC_init")
    axis.set_zlabel(output_keys[i])
plt.show()

