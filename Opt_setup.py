from csv import excel

import numpy as np
import math
from src.globals import Constants
from scipy import optimize

from src.Optimization import cost

normalization = Constants.get_normalization_factors()

if __name__ == "__main__":
    init = np.array(list(np.linspace(124734044, 2.22736184e+08, 5) / normalization["SMA"]) + list(np.linspace(0.31108, 3.77282462e-01, 5) / normalization["ECC"]) + list(np.linspace(0.3422, 9.08695610e-02, 5) / normalization["INC"]))
    # init = np.array([0.04595579, 0.23340204, 0.41669299, 0.55021038, 0.58833595, 0.48890959, 0.28930837, 0.10644652, 0.06069644, 0.27243051] + [0.54393869, 0.96396687, 0.96695896, 0.73015548, 0.43079695, 0.24132214, 0.22372922, 0.32957578, 0.50561767, 0.69861072] + [1.40745563, 1.01029051, 0.9049173,  0.94361087, 0.97864611, 0.86814603, 0.6047405,  0.3155662,  0.13360799, 0.19185069])

    bounds_SMA = np.array([(10000000, 300000000) for _ in range(10)]) / normalization["SMA"]
    bounds_ECC = np.array([(0.01, 0.99) for _ in range(10)]) / normalization["ECC"]
    bounds_INC = np.array([(0.001, 0.5 * math.pi) for _ in range(10)]) / normalization["INC"]

    bounds = tuple(list(bounds_SMA) + list(bounds_ECC) + list(bounds_INC))

    # sol = optimize.minimize(fun=cost.cost_function, x0=init, bounds=bounds, method='BFGS')
    # sol = optimize.shgo(func=cost.cost_function, bounds=bounds, workers=-1)
    sol = optimize.differential_evolution(func=cost.cost_function, bounds=bounds, workers=-1, strategy='randtobest1bin')

    print("##################")
    print()
    print("Solution: ")
    print()

    try:
        print(sol.x)
        print(sol.success)
        print(sol.message)
        print(sol)
    except:
        print("Error in solution print")
