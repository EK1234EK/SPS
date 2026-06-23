import math
import numpy as np
import ssl
from pyatmos import coesa76
ssl._create_default_https_context = ssl._create_unverified_context


from src.globals.Constants import get_globals

G, MY, KS_TOLERANCE, GRAV_CONST, EARTH_RADIUS = get_globals()


def get_coeff():
    coeff = {   25000: 	 [7249, 1.225E+00],
    30000: 	 [6349, 3.899E-02],
    40000: 	 [6682, 1.774E-02],
    50000: 	 [7554, 3.972E-03],
    60000: 	 [8382, 1.057E-03],
    70000: 	 [7714, 3.206E-03],
    80000: 	 [6549, 8.770E-05],
    90000: 	 [5799, 1.905E-05],
    100000:  [5382, 3.396E-06],
    110000:  [5877, 5.297E-07],
    120000:  [7263, 9.661E-08],
    130000:  [9473, 2.438E-08],
    140000:  [12636, 8.484E-09],
    150000:  [16149 ,3.845E-09],
    180000:  [22523, 2.070E-09],
    200000:  [29740, 5.464E-10],
    250000:  [37105, 2.789E-10],
    300000:  [45546, 7.248E-11],
    350000:  [53628, 2.418E-11],
    400000:  [53298, 9.518E-12],
    450000:  [58515, 3.725E-12],
    500000:  [60828, 1.585E-12],
    600000:  [63822, 6.967E-13],
    700000:  [71835, 1.454E-13],
    800000:  [88667, 3.614E-14],
    900000:  [124640, 1.170E-14],
    1000000: [181050, 5.245E-15],
	}
    return coeff

def get_aero_parameters():
    params = {
        "sigma_n": 0.8,
        "sigma_t": 0.8,
        "V_R": 0.05
    }
    return params

class Atmopshere:
    def __init__(self):
        self.atmo_coeff = get_coeff()  # On the form {init_altitude: [scale_height, density]}
        self.aero_params = get_aero_parameters()
        self.H = None
        self.scale_rho = None
        self.lower_altitude = None
        self.rho = None
        self.keys = list(self.atmo_coeff.keys())
        self.Cd = None

    def get_params(self, height):
        # key = self.keys[-1]
        for i, key in enumerate(self.keys):
            if height < key:
                if i == 0:
                    self.lower_altitude = 0
                else:
                    self.lower_altitude = self.keys[i-1]
                break
        params = self.atmo_coeff[key]
        self.H = params[0]
        self.scale_rho = params[1]

    def get_density(self, state: np.array):
        h = np.linalg.norm(state[0:3]) - EARTH_RADIUS
        self.get_params(height=h)
        self.rho = self.scale_rho * math.exp(-(h - self.lower_altitude) / self.H)

    def get_density_coesa76(self, state: np.array):
        h_km = (np.linalg.norm(state[0:3]) - EARTH_RADIUS) * 0.001
        if h_km > 1000:
            self.rho = 0
        else:
            coesa76_geom = coesa76([h_km])
            self.rho = coesa76_geom.rho[0]

    def get_Cd(self, vel: np.array, n: np.array):
        cos_aoa = np.dot(vel, n) / (np.linalg.norm(vel) * np.linalg.norm(n))
        # Armando, page 40
        self.Cd = 2 * (self.aero_params["sigma_t"] + self.aero_params["sigma_n"] * self.aero_params["V_R"] * abs(cos_aoa) + (2 - self.aero_params["sigma_n"] - self.aero_params["sigma_t"] * cos_aoa ** 2)) * abs(cos_aoa)

    def get_aero_acc(self, state: np.array, n: np.array, sigma: float):
        # Calling preparatory routines:
        self.get_density_coesa76(state=state)
        self.get_Cd(vel=state[3:6], n=n)
        # Armando, page 40
        acc = -0.5 * (self.rho / sigma) * np.linalg.norm(state[3:6]) * self.Cd * state[3:6]
        return acc

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    atmo = Atmopshere()
    state_list = [[x, 0, 0] for x in np.linspace(EARTH_RADIUS, EARTH_RADIUS + 10000000, 100)]
    state_list = [np.array(state) for state in state_list]

    state_mag = [np.linalg.norm(state) - EARTH_RADIUS for state in state_list]
    dens_list = np.zeros(len(state_mag))
    for k, state in enumerate(state_list):
        atmo.get_density(state=state)
        dens_list[k] = atmo.rho

    # The pyatmos option
    # import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    # from pyatmos import coesa76

    dens_list_acc = np.zeros(len(state_mag))
    for k, state in enumerate(state_list):
        coesa76_geom = coesa76([(np.linalg.norm(state) - EARTH_RADIUS)*0.001])
        rho = coesa76_geom.rho[0]
        print(rho)
        if rho == 0:
            dens_list_acc[k] = None
        else:
            dens_list_acc[k] = rho



    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(state_mag, [math.log(d, 10) for d in dens_list], label="Exponential Armando")
    ax.plot(state_mag, [math.log(d_c, 10) for d_c in dens_list_acc], label="coesa76")
    ax.set_title("Decadic logarithm of density")
    ax.set_xlabel("Altitude [m]")
    ax.set_ylabel("Dec. log. of density")
    ax.legend()

    ax_2 = fig.add_subplot(212)
    ax_2.plot(state_mag, dens_list - dens_list_acc, label="Error")
    ax_2.set_title("Density error")
    ax_2.set_xlabel("Altitude [m]")
    ax_2.set_ylabel("Density error [kg / m^3]")
    ax_2.legend()
    plt.show()


