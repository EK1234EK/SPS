import matplotlib.pyplot as plt
import pandas
import numpy as np
import pandas as pd
from numpy.ma.core import argmin

from  src.guidance import reference_trajectory as rt
from src.globals import Constants

normalization = Constants.get_normalization_factors()

def get_solution_arrays(pattern: str, base_df):
    out_array = [[]]
    for key in base_df.keys():
        if pattern in key and not out_array[0]:
            out_array[0] = base_df[key].tolist()
        elif pattern in key:
            out_array = out_array + [base_df[key].tolist()]
    return np.array(out_array)

def plot_solutions(key, axis, sorted, sample_time, min_cost_idx=None, min_cost=None):
    idx = ["SMA", "ECC", "INC"].index(key)
    for i in range(len(sorted[0, :])):
        # for i in range(100):
        solution = sorted[:, i]
        solution_res = []
        rt_base = rt.steering_track(knot_points={key: solution, "t": np.linspace(sample_time[0], sample_time[-1], len(sorted[:,0]))})
        for t in sample_time:
            setpoint = rt_base.guidance_setpoint(time=t)
            solution_res.append(setpoint[idx])
        axis.plot(sample_time, solution_res, color=[0, 0.5, 0.5], alpha=0.1)
        if key == "SMA":
            axis.set_title(key + ", normalized at " + str(normalization[key]) + " m")
        else:
            axis.set_title(key + ", normalized at " + str(normalization[key]))
        axis.set_xlabel("Time [s]")
        axis.grid(True)
        axis.minorticks_on()

    if min_cost_idx:
        solution = sorted[:, min_cost_idx]
        print(key, ": ", solution)
        solution_res = []
        rt_base = rt.steering_track(knot_points={key: solution, "t": np.linspace(sample_time[0], sample_time[-1], len(sorted[:, 0]))})
        for t in sample_time:
            setpoint = rt_base.guidance_setpoint(time=t)
            solution_res.append(setpoint[idx])
        if min_cost:
            axis.plot(sample_time, solution_res, color=[1, 0, 1], label="Optimal at " + str(round(min_cost / (24*3600), 3)) + " days")
        else:
            axis.plot(sample_time, solution_res, color=[1, 0, 1], label="Optimal")
    axis.legend()

data = pd.read_csv("iterations.csv")
sorted_SMA = get_solution_arrays(pattern="SMA", base_df=data)
sorted_ECC = get_solution_arrays(pattern="ECC", base_df=data)
sorted_INC = get_solution_arrays(pattern="INC", base_df=data)

static_cost = data["cost_static"].tolist()
integral_cost = -np.array(data["cost_integral"].tolist())
total_cost = data["cost_total"].tolist()

ratio = np.zeros(len(static_cost))

min_idx = argmin(total_cost)

for i in range(len(static_cost)):
    if static_cost[i] >= 10**10:
        ratio[i] = None
        static_cost[i] = None
        total_cost[i] = None
        integral_cost[i] = None
    else:
        ratio[i] = integral_cost[i] / (total_cost[i])

sample_time = np.linspace(-300 * 24 * 3600, 0, 500)

# Plot the solution space

fig_1 = plt.figure()
ax_1 = fig_1.add_subplot(221)
ax_2 = fig_1.add_subplot(222)
ax_3 = fig_1.add_subplot(223)
plot_solutions(key="SMA", axis=ax_1, sorted=sorted_SMA, sample_time=sample_time, min_cost_idx=min_idx, min_cost=min(t for t in total_cost if t is not None))
plot_solutions(key="ECC", axis=ax_2, sorted=sorted_ECC, sample_time=sample_time, min_cost_idx=min_idx, min_cost=min(t for t in total_cost if t is not None))
plot_solutions(key="INC", axis=ax_3, sorted=sorted_INC, sample_time=sample_time, min_cost_idx=min_idx, min_cost=min(t for t in total_cost if t is not None))

# Plot the cost
iterations = np.arange(len(static_cost))
fig_2 = plt.figure()
ax_1 = fig_2.add_subplot(321)
ax_2 = fig_2.add_subplot(322)
ax_3 = fig_2.add_subplot(323)
ax_4 = fig_2.add_subplot(324)
ax_5 = fig_2.add_subplot(325)
ax_1.plot(iterations, static_cost, color=[0, 0.5, 0.5])
ax_2.plot(iterations, integral_cost, color=[0.5, 0, 0.5])
ax_3.plot(iterations, total_cost, color=[0, 0, 1])

ax_1.set_title("Static cost")
ax_2.set_title("Integral cost")
ax_3.set_title("Total cost")

ax_4.plot(iterations, static_cost, color=[0, 0.5, 0.5], label="Static cost [s]")
ax_4.plot(iterations, integral_cost, color=[0.5, 0, 0.5], label="Integral cost [s]")
ax_4.plot(iterations, total_cost, color=[0, 0, 1], label="Total cost [s]")
ax_4.set_xlabel("Iteration")
ax_4.set_title("Overview")
ax_4.legend()

ax_5.plot(iterations, ratio, color=[0, 0.5, 0.5], label="Ratio integral / total cost")
ax_5.set_title("Ratio of integral to total cost")
ax_5.set_xlabel("Iteration")

plt.show()
