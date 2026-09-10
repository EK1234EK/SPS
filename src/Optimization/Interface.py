import copy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from numba.core.ir_utils import transfer_scope

class Interface:
    def __init__(self, discrete_interface: pd.DataFrame, interface_space: list, target_dimension: str = "t_s"):
        self.discrete_interface = discrete_interface
        self.interface_space = interface_space
        self.target_dimension = target_dimension

        self.reduce_discrete_dataset()
        self.bounds = self.get_convex_rectangle()

    def filter_interface(self, bounds: dict[str: tuple[float, float]]):
        new_interface = pd.DataFrame(columns=self.discrete_interface.keys())
        rows = []
        for c in self.discrete_interface.index.values.tolist():
            get_row = True
            for i, key in enumerate(bounds.keys()):
                if key in self.discrete_interface.keys():
                    if not (bounds[key][0] < self.discrete_interface[key][c] < bounds[key][1]):
                        get_row = False

            if get_row:
                row = self.discrete_interface.iloc[[c]].values.flatten().tolist()
                rows.append(row)

        for r in range(len(rows)):
            new_interface.loc[r] = rows[r]

        self.discrete_interface = new_interface

    def interface_interpolation(self, eval_points: list, method="linear", fill_val=0):
        points = []
        values = self.discrete_interface[self.target_dimension].tolist()
        for key in self.discrete_interface.keys():
            if key != self.target_dimension:
                points.append(self.discrete_interface[key].tolist())  # TODO  - wrong shape, all one axis

        points = list(np.transpose(np.array(points)))

        interp_vals = scipy.interpolate.griddata(points=points, values=values, xi=eval_points, method=method,
                                                 fill_value=fill_val, rescale=True)
        return interp_vals

    def get_convex_rectangle(self):
        # We just assume that the interface subspace is a hyper-rectangle. This is obviously very wrong,
        # but I just leave it as a problem for future me
        bounds = dict()
        for key in self.discrete_interface.keys():
            bounds[key] = [min(self.discrete_interface[key]), max(self.discrete_interface[key])]
        return bounds

    def reduce_discrete_dataset(self):
        new_df = pd.DataFrame()
        for key in self.discrete_interface.keys():
            if key in self.interface_space:
                new_df[key] = self.discrete_interface[key]
        self.discrete_interface = new_df

class plotter:
    def __init__(self, sampling):
        self.sampling = sampling
        interface_df = pd.read_csv("../../scenarios/Legacy/Interface_sigma_04.csv")

        """transfer_interface = Interface(discrete_interface=interface_df, interface_space=["SMA", "ECC", "INC", "r_init"], target_dimension="r_init")
        transfer_interface.filter_interface(bounds={"r_init": (0, 7000000)})
        eval_points = [[187313538.85384998, 0.45943407516754026, 0.17404362669030468]]

        interface_vals = transfer_interface.interface_interpolation(eval_points=eval_points, fill_val=69)
        pass"""

        n_samp = self.sampling
        offset_extend = 0.1

        dof = ["SMA", "ECC", "INC"]

        self.transfer_interface = Interface(discrete_interface=interface_df, interface_space=dof + ["t_s"],
                                            target_dimension="t_s")
        self.transfer_interface.filter_interface(bounds={"r_init": (0, 7000000)})

        self.free_dims = ["SMA", "INC"]
        self.fixed_dims = ["ECC"]

        bounds_all = self.transfer_interface.get_convex_rectangle()
        bounds = dict()
        for key in self.free_dims + self.fixed_dims:
            bounds[key] = bounds_all[key]

        offset_1 = (bounds[self.free_dims[0]][1] - bounds[self.free_dims[0]][0]) * offset_extend
        offset_2 = (bounds[self.free_dims[1]][1] - bounds[self.free_dims[1]][0]) * offset_extend

        x_samp = np.linspace(bounds[self.free_dims[0]][0] - offset_1, bounds[self.free_dims[0]][1] + offset_1,
                             n_samp)
        y_samp = np.linspace(bounds[self.free_dims[1]][0] - offset_2, bounds[self.free_dims[1]][1] + offset_2,
                             n_samp)

        self.X, self.Y = np.meshgrid(x_samp, y_samp)
        self.sol_matrix = copy.deepcopy(self.X)
        self.list_of_eval_points = []
        self.list_x = []
        self.list_y = []

        for ix, x in enumerate(x_samp):
            for iy, y in enumerate(y_samp):

                eval_point = [_ for _ in range(len(dof))]

                for di, d in enumerate(dof):
                    if d in self.fixed_dims:
                        eval_point[di] = bounds[d][0] * 0.5 + bounds[d][1] * 0.5

                    if d == self.free_dims[0]:
                        eval_point[di] = x
                    elif d == self.free_dims[1]:
                        eval_point[di] = y
                eval_point = [eval_point]
                self.list_of_eval_points.append(eval_point)
                self.list_x.append(ix)
                self.list_y.append(iy)

                # interface_val = transfer_interface.interface_interpolation(eval_points=eval_point, fill_val=100000000)[0]
                # print(interface_val)
                # sol_matrix[ix][iy] = interface_val

    def eval_set(self, i):
        interface_val = self.transfer_interface.interface_interpolation(eval_points=self.list_of_eval_points[i],
                                                                        fill_val=0)[0]
        # print(interface_val)
        idx = self.list_x[i]
        idy = self.list_y[i]
        # sol_matrix[idx][idy] = interface_val
        return (interface_val, idx, idy)

    def run_parallel(self):
        lst_val = Pool.map(self.eval_set, range(len(self.list_of_eval_points)))
        return lst_val

if __name__ == "__main__":
    import multiprocessing

    Pool = multiprocessing.Pool(12)
    import matplotlib as mpl

    mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'

    sampling = 300
    sol_matrix = np.zeros([sampling, sampling])

    plotter_class = plotter(sampling=sampling)
    lst_vals = plotter_class.run_parallel()
    for val in lst_vals:
        if val[0] != 0:
            sol_matrix[val[1]][val[2]] = val[0]
        else:
            sol_matrix[val[1]][val[2]] = None

    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(121)
    ax_2 = fig_1.add_subplot(122, projection="3d")
    # ax_1.imshow(sol_matrix)
    ax_1.imshow(sol_matrix, cmap="jet")
    ax_2.plot_surface(plotter_class.X, plotter_class.Y, sol_matrix, cmap="jet")
    ax_2.set_xlabel(plotter_class.free_dims[0])
    ax_2.set_ylabel(plotter_class.free_dims[1])
    plt.show()
