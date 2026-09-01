import matplotlib.pyplot as plt
import pandas
import numpy as np
import pandas as pd
from numpy.ma.core import argmin
from scipy.stats import randint
import random

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
        if i == 2042:
            pass
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
        for idx_cost in min_cost_idx:
            solution = sorted[:, idx_cost]
            print(key, ": ", solution)
            solution_res = []
            rt_base = rt.steering_track(knot_points={key: solution, "t": np.linspace(sample_time[0], sample_time[-1], len(sorted[:, 0]))})
            for t in sample_time:
                setpoint = rt_base.guidance_setpoint(time=t)
                solution_res.append(setpoint[idx])
            if min_cost:
                for cost in min_cost:
                    axis.plot(sample_time, solution_res, color=[1, 0, 1], label="Optimal at " + str(round(cost / (24*3600), 3)) + " days")
            else:
                axis.plot(sample_time, solution_res, color=[1, 0, 1], label="Optimal")
    axis.legend()

if __name__ == "__main__":

    data = pd.read_csv("iterations.csv")
    sorted_SMA = get_solution_arrays(pattern="SMA", base_df=data)
    sorted_ECC = get_solution_arrays(pattern="ECC", base_df=data)
    sorted_INC = get_solution_arrays(pattern="INC", base_df=data)
    time_offset = get_solution_arrays(pattern="time_start", base_df=data)[0] * normalization["time_offset"]

    static_cost = data["cost_static"].tolist()
    integral_cost = -np.array(data["cost_integral"].tolist())
    # integral_cost_correction = integral_cost - np.array(time_offset)
    total_cost = list(np.array(data["cost_total"]) - np.array(time_offset))
    total_cost_base = list(np.array(data["cost_total"]))

    ratio = np.zeros(len(static_cost))

    # min_idx = argmin(total_cost)
    nsol = 1
    min_cost = sorted(total_cost)[:nsol]
    min_idx = [total_cost.index(c) for c in min_cost]
    print("Minimum cost index: ", min_idx, " at cost ", [total_cost[i] for i in min_idx])

    for i in range(len(static_cost)):
        if static_cost[i] >= 10**10:
            ratio[i] = None
            static_cost[i] = None
            total_cost[i] = None
            integral_cost[i] = None
        else:
            ratio[i] = integral_cost[i] / (total_cost[i])

    sample_time = np.linspace(-30 * 24 * 3600, 0, 500)

    # Plot the solution space

    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(221)
    ax_2 = fig_1.add_subplot(222)
    ax_3 = fig_1.add_subplot(223)
    ax_4 = fig_1.add_subplot(224)
    """plot_solutions(key="SMA", axis=ax_1, sorted=sorted_SMA, sample_time=sample_time, min_cost_idx=min_idx, min_cost=min(t for t in total_cost if t is not None))
    plot_solutions(key="ECC", axis=ax_2, sorted=sorted_ECC, sample_time=sample_time, min_cost_idx=min_idx, min_cost=min(t for t in total_cost if t is not None))
    plot_solutions(key="INC", axis=ax_3, sorted=sorted_INC, sample_time=sample_time, min_cost_idx=min_idx, min_cost=min(t for t in total_cost if t is not None))"""

    plot_solutions(key="SMA", axis=ax_1, sorted=sorted_SMA, sample_time=sample_time, min_cost_idx=min_idx, min_cost=min_cost)
    plot_solutions(key="ECC", axis=ax_2, sorted=sorted_ECC, sample_time=sample_time, min_cost_idx=min_idx, min_cost=min_cost)
    plot_solutions(key="INC", axis=ax_3, sorted=sorted_INC, sample_time=sample_time, min_cost_idx=min_idx, min_cost=min_cost)
    ax_4.scatter(total_cost, time_offset, label="Time offset", color=[0, 0.5, 0.5], marker=".")
    ax_4.set_ylabel("Time offset [s]")
    ax_4.set_xlabel("Total cost")

    for i, idx in enumerate(min_idx):
        print("Time offset ", i+1, ": ", time_offset[idx])
        ax_4.scatter(total_cost[idx], time_offset[idx], color=[1, 0, 1])

    # Plot the cost
    iterations = np.arange(len(static_cost))
    fig_2 = plt.figure()
    ax_1 = fig_2.add_subplot(321)
    ax_2 = fig_2.add_subplot(322)
    ax_3 = fig_2.add_subplot(323)
    ax_4 = fig_2.add_subplot(324)
    ax_5 = fig_2.add_subplot(325)
    ax_1.scatter(iterations, static_cost, color=[0, 0.5, 0.5], marker=".")
    ax_2.scatter(iterations, integral_cost, color=[0.5, 0, 0.5], marker=".")
    ax_3.scatter(iterations, total_cost, color=[0, 0, 1], marker=".")

    ax_1.set_title("Static cost")
    ax_2.set_title("Integral cost")
    ax_3.set_title("Total cost")

    ax_4.scatter(iterations, static_cost, color=[0, 0.5, 0.5], label="Static cost [s]", marker=".")
    ax_4.scatter(iterations, integral_cost, color=[0.5, 0, 0.5], label="Integral cost [s]", marker=".")
    ax_4.scatter(iterations, total_cost, color=[0, 0, 1], label="Total cost [s]", marker=".")
    ax_4.set_xlabel("Iteration")
    ax_4.set_title("Overview")
    ax_4.legend()

    ax_5.scatter(iterations, ratio, color=[0, 0.5, 0.5], label="Ratio integral / total cost", marker=".")
    ax_5.set_title("Ratio of integral to total cost")
    ax_5.set_xlabel("Iteration")

    # Plot intercept parameters
    intercept_SMA = get_solution_arrays(pattern="a_intercept", base_df=data)[0]
    intercept_ECC= get_solution_arrays(pattern="e_intercept", base_df=data)[0]
    intercept_INC = get_solution_arrays(pattern="i_intercept", base_df=data)[0]
    fig_3 = plt.figure()
    ax_1 = fig_3.add_subplot(221)
    ax_2 = fig_3.add_subplot(222)
    ax_3 = fig_3.add_subplot(223)

    ax_1.set_ylabel("Total cost")
    ax_1.set_xlabel("Intercept SMA")

    ax_2.set_ylabel("Total cost")
    ax_2.set_xlabel("Intercept ECC")

    ax_3.set_ylabel("Total cost")
    ax_3.set_xlabel("Intercept INC")

    ax_1.scatter(intercept_SMA, total_cost, color=[0, 0.5, 0.5], marker=".", label="Intercept SMA")
    ax_2.scatter(intercept_ECC, total_cost, color=[0, 0.5, 0.5], marker=".", label="Intercept ECC")
    ax_3.scatter(intercept_INC, total_cost, color=[0, 0.5, 0.5], marker=".", label="Intercept INC")
    fig_3.legend()

    print("Intercept parameters:")
    for i in min_idx:
        print("Intercept SMA: ", intercept_SMA[i])
        print("Intercept ECC: ", intercept_ECC[i])
        print("Intercept INC: ", intercept_INC[i])

    # Plot initial parameters

    r_init = get_solution_arrays(pattern="r_init", base_df=data)[0]
    i_init = get_solution_arrays(pattern="i_init", base_df=data)[0]
    solar_phasing = get_solution_arrays(pattern="solar_phasing", base_df=data)[0]
    propagation_cutoff_a = get_solution_arrays(pattern="propagation_cutoff_a", base_df=data)[0]
    fig_4 = plt.figure()
    ax_1 = fig_4.add_subplot(221)
    ax_2 = fig_4.add_subplot(222)
    ax_3 = fig_4.add_subplot(223)
    ax_4 = fig_4.add_subplot(224)

    ax_1.set_ylabel("Total cost")
    ax_1.set_xlabel("r_init")

    ax_2.set_ylabel("Total cost")
    ax_2.set_xlabel("i_init")

    ax_3.set_ylabel("Total cost")
    ax_3.set_xlabel("solar_phasing")

    ax_4.set_ylabel("Total cost")
    ax_4.set_xlabel("propagation_cutoff_a")

    ax_1.scatter(r_init, total_cost, color=[0, 0.5, 0.5], marker=".", label="Initial radius")
    ax_2.scatter(i_init, total_cost, color=[0, 0.5, 0.5], marker=".", label="Initial inclination")
    ax_3.scatter(solar_phasing, total_cost, color=[0, 0.5, 0.5], marker=".", label="Solar phasing")
    ax_4.scatter(propagation_cutoff_a, total_cost, color=[0, 0.5, 0.5], marker=".", label="Propagation cutoff")
    fig_4.legend()

    print("Initial parameters: ")
    for i in min_idx:
        print("r_init: ", r_init[i])
        print("i_init ", i_init[i])
        print("solar phasing: ", solar_phasing[i])
        print("Propagation cutoff SMA: ", propagation_cutoff_a[i])

    plt.show()
