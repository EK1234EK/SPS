from csv import excel

import numpy as np
import math
from src.globals import Constants
from scipy import optimize

from src.Optimization import cost

normalization = Constants.get_normalization_factors()

if __name__ == "__main__":
    init = np.array(list(np.linspace(124734044, 2.22736184e+08, 5) / normalization["SMA"]) + list(np.linspace(0.31108, 3.77282462e-01, 5) / normalization["ECC"]) + list(np.linspace(0.3422, 9.08695610e-02, 5) / normalization["INC"]))

    bounds_SMA = np.array([(10000000, 300000000) for _ in range(5)]) / normalization["SMA"]
    bounds_ECC = np.array([(0.01, 0.99) for _ in range(5)]) / normalization["ECC"]
    bounds_INC = np.array([(0.001, 0.5 * math.pi) for _ in range(5)]) / normalization["INC"]

    bounds = tuple(list(bounds_SMA) + list(bounds_ECC) + list(bounds_INC))

    # sol = optimize.minimize(fun=cost.cost_function, x0=init, bounds=bounds, method='BFGS')
    sol = optimize.shgo(func=cost.cost_function, bounds=bounds, workers=-1)

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
