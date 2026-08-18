import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import label


class steering_track:
    def __init__(self, knot_points: (dict | None)=None):
        # Collocation points: dict with a subset pf the keys a, e, i, RAAN, APERI, TAEPO, t --> with reference time
        # Dict includes list, even for just a single point --> dict(key: list)

        self.knot_points = knot_points
        self.all_params = ["SMA", "ECC", "INC", "RAAN", "APERI", "TAEPO"]
        self.set_params = list(self.knot_points.keys())
        self.set_params.remove("t")

        if len(self.knot_points["t"]) > 1:
            self.cs = self.get_cs()

    def get_cs(self):
        cs = dict()
        for p in self.all_params:
            if p in self.set_params:
                cs[p] = CubicSpline(x=self.knot_points["t"], y=self.knot_points[p])
        return cs

    def guidance_setpoint(self, time):
        setpoint = np.array([None for _ in range(len(self.all_params))])
        if len(self.knot_points["t"]) == 1:
            # Constant guidance setpoint
            for i, key in enumerate(self.all_params):
                if key in self.set_params:
                    setpoint[i] = self.knot_points[key][0]
            return setpoint

        elif len(self.knot_points) > 1:
            if time < self.knot_points["t"][0]:
                for i, key in enumerate(self.all_params):
                    if key in self.set_params:
                        setpoint[i] = self.knot_points[key][0]
            elif time > self.knot_points["t"][-1]:
                for i, key in enumerate(self.all_params):
                    if key in self.set_params:
                        setpoint[i] = self.knot_points[key][-1]
            else:
                for i, key in enumerate(self.all_params):
                    if key in self.set_params:
                        inp_val = self.cs[key](np.array([time]))
                        setpoint[i] = inp_val[0]
            return setpoint
        else:
            return setpoint

if __name__ == "__main__":
    import random
    ncol = 5
    # knot_points = {"SMA": [0.04595579, 0.45652057, 0.5573711, 0.1440098, 0.27243051], "ECC": [0.54393869, 0.92331547, 0.31304721, 0.29445348, 0.69861072], "INC": [1.40745563, 0.90685494, 0.94862776, 0.38300299, 0.19185069], "t": np.arange(-ncol, 0)}
    knot_points = {"SMA": [0.04595579, 0.23340204, 0.41669299, 0.55021038, 0.58833595, 0.48890959, 0.28930837, 0.10644652, 0.06069644, 0.27243051],
                   "ECC": [0.54393869, 0.96396687, 0.96695896, 0.73015548, 0.43079695, 0.24132214, 0.22372922, 0.32957578, 0.50561767, 0.69861072],
                   "INC": [1.40745563, 1.01029051, 0.9049173,  0.94361087, 0.97864611, 0.86814603, 0.6047405,  0.3155662,  0.13360799, 0.19185069],
                   "t": np.linspace(-100 * 24 * 3600, 0, 10)}
    target_track = steering_track(knot_points=knot_points)

    times = np.linspace(knot_points["t"][0], knot_points["t"][-1], 10)
    SMA_track = np.zeros(len(times))
    ECC_track = np.zeros(len(times))
    INC_track = np.zeros(len(times))

    for t, time in enumerate(times):
        setpoint = target_track.guidance_setpoint(time=time)
        SMA_track[t] = setpoint[0]
        ECC_track[t] = setpoint[1]
        INC_track[t] = setpoint[2]

    pass
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times, SMA_track, color=[1, 0, 0], label="SMA")
    ax.plot(times, ECC_track, color=[0, 1, 0], label="ECC")
    ax.plot(times, INC_track, color=[0, 0, 1], label="INC")
    for key in knot_points.keys():
        if key != "t":
            ax.scatter(knot_points["t"], knot_points[key], label=key + " col point")
    ax.set_xlabel("Time []")
    ax.set_ylabel("Parameter value []")
    ax.set_title("Example of knot point and spline")
    ax.legend()
    plt.show()

