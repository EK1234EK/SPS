import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Plot the shit
output_keys = ["SMA", "ECC", "INC", "t_s"]
input_keys = ["r_init", "INC_init", "solar_phasing", "propagation_cutoff_SMA"]

fig = plt.figure()
axes = []
for i in range(len(input_keys)):
    axis = fig.add_subplot(2, 2, i+1)
    axis.set_title(input_keys[i])
    for j in range(len(output_keys)):
        axis.scatter(normalized_data[input_keys[i]].tolist(), normalized_data[output_keys[j]].tolist(), label=output_keys[j])
        axis.set_xlabel(input_keys[i])
        axis.set_ylabel("Output")
    axes.append(axis)
axes[0].legend()

plt.show()

