from csv import excel

import numpy as np
import math
from src.globals import Constants
from scipy import optimize

from src.Optimization import cost

normalization = Constants.get_normalization_factors()

"""SMA = [0.3451010210605255, 0.7343221579635179, 0.2408057331982619, 0.5380917695693918, 0.1369039452556938, 0.5050843907208235, 0.4650072807272808, 0.6755022808231158, 0.4823315736635267, 0.2928294370131951]
ECC = [0.540108303729281, 0.7550813911086498, 0.5025371080356382, 0.7804030566571856, 0.1490874361194686, 0.9494793376087238, 0.4458715666636607, 0.1766284570039121, 0.908519502514348, 0.355569136754177]
INC = [0.5738214819347085, 1.4298811454999305, 1.4061870568091237, 0.959828100733754, 0.0292528781279597, 0.1833147629578032, 0.1298259000934447, 1.4825489355029124, 0.5405643865953869, 0.472364978275073]
dof = SMA + ECC + INC + [2174659.6485664207 / normalization["time_offset"]]
tof = cost.cost_function(dof, True)
print(tof)
exit()"""

if __name__ == "__main__":
    init = np.array(list(np.linspace(124734044, 2.22736184e+08, 5) / normalization["SMA"]) + list(np.linspace(0.31108, 3.77282462e-01, 5) / normalization["ECC"]) + list(np.linspace(0.3422, 9.08695610e-02, 5) / normalization["INC"]))
    # init = np.array([0.04595579, 0.23340204, 0.41669299, 0.55021038, 0.58833595, 0.48890959, 0.28930837, 0.10644652, 0.06069644, 0.27243051] + [0.54393869, 0.96396687, 0.96695896, 0.73015548, 0.43079695, 0.24132214, 0.22372922, 0.32957578, 0.50561767, 0.69861072] + [1.40745563, 1.01029051, 0.9049173,  0.94361087, 0.97864611, 0.86814603, 0.6047405,  0.3155662,  0.13360799, 0.19185069])

    bounds_SMA = np.array([(10000000, 300000000) for _ in range(10)]) / normalization["SMA"]
    bounds_ECC = np.array([(0.01, 0.99) for _ in range(10)]) / normalization["ECC"]
    bounds_INC = np.array([(0.001, 0.5 * math.pi) for _ in range(10)]) / normalization["INC"]

    bounds = tuple(list(bounds_SMA) + list(bounds_ECC) + list(bounds_INC) + [(-1, 1)])

    # sol = optimize.minimize(fun=cost.cost_function, x0=init, bounds=bounds, method='BFGS')
    # sol = optimize.shgo(func=cost.cost_function, bounds=bounds, workers=-1)
    sol = optimize.differential_evolution(func=cost.cost_function, bounds=bounds, workers=1, strategy='rand1bin')

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
