import numpy as np
import pandas as pd
from numpy.ma.core import argmin
from setuptools.installer import fetch_build_egg

import Interface

def get_interface_bc(intercept_parameters):
    itf = Interface.Interface(discrete_interface=pd.read_csv("../../scenarios/Legacy/Interface_sigma_04.csv"), interface_states=["SMA", "ECC", "INC", "r_init", "INC_init", "solar_phasing", "propagation_cutoff_SMA", "t_s"])
    keys = ["SMA", "ECC", "INC"]
    fit_idx_multi = []

    for i in range(3):
        itc = np.ones(len(itf.discrete_interface[keys[i]])) * intercept_parameters[i]
        diff = itc - np.array(itf.discrete_interface[keys[i]])
        diff = [abs(d) for d in diff]
        fit_idx = argmin(diff)
        fit_idx_multi.append(fit_idx)
    pass


if __name__ == "__main__":
    intercept_parameters = [184944021.01895648, 0.3731525552093156, 0.21786674439400172, 0.05232783642950021, 0.23503935113627727, 4.2169267879241445]
    get_interface_bc(intercept_parameters)
    pass