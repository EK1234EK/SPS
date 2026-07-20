import matplotlib.pyplot as plt
import numpy as np
import math


def bisecting(fun, lb, ub, x_tol):
    """b_1 = 0.05
    b_2 = 1.9
    b_3 = 1.2666"""
    n_eval = 0
    m = 0.5 * (lb + ub)

    """eval_arr = np.linspace(lb, ub, 100)
    fun_arr = np.array([fun(a) for a in eval_arr])
    int_fun_arr = np.array([np.trapezoid(fun_arr[0:i]) for i in range(len(eval_arr))])
    denom_arr = np.array([3 * b_2 * math.cos(a) ** 3 + 2 * b_3 * math.cos(a) ** 2 - 2 * b_2 * math.cos(
        a) - b_3 for a in eval_arr])  # TODO Is this really an exponent of 3, or a typo?
    
    fun_arr = fun_arr / np.linalg.norm(max(fun_arr))
    int_fun_arr = int_fun_arr / np.linalg.norm(max(int_fun_arr))
    denom_arr = denom_arr / np.linalg.norm(max(denom_arr))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(eval_arr, fun_arr, label="Function")
    ax.scatter(eval_arr, int_fun_arr, label="Integral")
    ax.scatter(eval_arr, denom_arr, label="Denominator")
    ax.legend()
    plt.show()"""

    while abs(ub - lb) > x_tol:
        a = fun(lb)
        b = fun(ub)
        m = 0.5 * (lb + ub)
        m_fun = fun(m)
        n_eval += 1

        if m_fun * a < 0:
            ub = m
        else:
            lb = m
        pass
    print(m, "  - ", n_eval)
    return m, n_eval
