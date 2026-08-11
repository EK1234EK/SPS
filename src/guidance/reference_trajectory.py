import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import label


class steering_track:
    def __init__(self, collocation_points: (dict | None)=None):
        # Collocation points: dict with a subset pf the keys a, e, i, RAAN, APERI, TAEPO, t --> with reference time
        # Dict includes list, even for just a single point --> dict(key: list)

        self.collocation_points = collocation_points
        self.all_params = ["SMA", "ECC", "INC", "RAAN", "APERI", "TAEPO"]
        self.set_params = list(self.collocation_points.keys())
        self.set_params.remove("t")

        if len(self.collocation_points["t"]) > 1:
            self.cs = self.get_cs()

    def get_cs(self):
        cs = dict()
        for p in self.all_params:
            if p in self.set_params:
                cs[p] = CubicSpline(x=self.collocation_points["t"], y=self.collocation_points[p])
        return cs

    def guidance_setpoint(self, time):
        setpoint = np.array([None for _ in range(len(self.all_params))])
        if len(self.collocation_points["t"]) == 1:
            # Constant guidance setpoint
            for i, key in enumerate(self.all_params):
                if key in self.set_params:
                    setpoint[i] = self.collocation_points[key][0]
            return setpoint

        elif len(self.collocation_points) > 1:
            if time < self.collocation_points["t"][0]:
                for i, key in enumerate(self.all_params):
                    if key in self.set_params:
                        setpoint[i] = self.collocation_points[key][0]
            elif time > self.collocation_points["t"][-1]:
                for i, key in enumerate(self.all_params):
                    if key in self.set_params:
                        setpoint[i] = self.collocation_points[key][-1]
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
    ncol = 10
    collolcation_points = {"SMA": [random.randint(-10, 10) for _ in range(ncol)], "ECC": [random.randint(-10, 10) for _ in range(ncol)], "t": np.arange(-ncol, 0)}
    target_track = steering_track(collocation_points=collolcation_points)

    times = np.linspace(collolcation_points["t"][0] - 2, collolcation_points["t"][-1] + 2, 10000)
    SMA_track = np.zeros(len(times))
    ECC_track = np.zeros(len(times))

    for t, time in enumerate(times):
        setpoint = target_track.guidance_setpoint(time=time)
        SMA_track[t] = setpoint[0]
        ECC_track[t] = setpoint[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times, SMA_track, color=[1, 0, 0], label="SMA")
    ax.plot(times, ECC_track, color=[0, 1, 0], label="ECC")
    for key in collolcation_points.keys():
        if key != "t":
            ax.scatter(collolcation_points["t"], collolcation_points[key], label=key + " col point")
    ax.legend()
    plt.show()

