import pandas as pd
import numpy as np
import scipy

class Interface:
    def __init__(self, discrete_interface: pd.DataFrame, interface_space: list, target_dimension: str="t_s"):
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
            if key in self.interface_space:
                new_df[key] = self.discrete_interface[key]
        self.discrete_interface = new_df

if __name__ == "__main__":
    interface_df = pd.read_csv("../../scenarios/Legacy/Interface_sigma_04.csv")

    transfer_interface = Interface(discrete_interface=interface_df, interface_space=["SMA", "ECC", "INC", "r_init"], target_dimension="r_init")
    transfer_interface.filter_interface(bounds={"r_init": (0, 7000000)})
    eval_points = [[187313538.85384998, 0.45943407516754026, 0.17404362669030468]]

    interface_vals = transfer_interface.interface_interpolation(eval_points=eval_points, fill_val=69)
    pass