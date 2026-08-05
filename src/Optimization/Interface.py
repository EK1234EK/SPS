import pandas as pd
import numpy as np
import scipy
from numba.core.ir_utils import transfer_scope


class Interface:
    def __init__(self, discrete_interface: pd.DataFrame, interface_states: list):
        self.discrete_interface = discrete_interface
        self.interface_states = interface_states

        self.reduce_discrete_dataset()
        self.bounds = self.get_convex_rectangle()

    def interface_interpolation(self, eval_points: list, method="linear", fill_val=0):
        points = []
        values = self.discrete_interface["t_s"].tolist()
        for key in self.discrete_interface.keys():
            if key != "t_s":
                points.append(self.discrete_interface[key].tolist())  # TODO  - wrong shape, all one axis

        points = list(np.transpose(np.array(points)))

        interp_vals = scipy.interpolate.griddata(points=points, values=values, xi=eval_points, method=method, fill_value=fill_val, rescale=True)
        return interp_vals

    def get_convex_rectangle(self):
        # We just assume that the interface subspace is a hyper-rectangle. This is obviously very wrong,
        # but I just leave it as a problem for future m
        bounds = dict()
        for key in self.discrete_interface.keys():
            bounds[key] = [min(self.discrete_interface[key]), max(self.discrete_interface[key])]
        return bounds

    def reduce_discrete_dataset(self):
        new_df = pd.DataFrame()
        for key in self.discrete_interface.keys():
            if key in self.interface_states:
                new_df[key] = self.discrete_interface[key]
        self.discrete_interface = new_df

if __name__ == "__main__":
    interface_df = pd.read_csv("../../scenarios/Legacy/Interface_sigma_04.csv")

    transfer_interface = Interface(discrete_interface=interface_df, interface_states=["SMA", "ECC", "INC", "t_s"])

    eval_points = [[90059841.97668832, 0.2167643850940429, 99432709.64873822],
                   [100059841.97668832, 0.2567643850940429, 89432709.64873822]]

    interface_vals = transfer_interface.interface_interpolation(eval_points=eval_points)
    pass