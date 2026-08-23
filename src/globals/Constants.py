import math

def get_globals():

    G = 9.81
    # MY = 3.98589*10**14
    MY = 1.3271011*10**20
    KS_TOLERANCE = 0.0000001
    GRAV_CONST = 6.67430*10**(-11)
    EARTH_RADIUS = 6378000
    OBLIQUITY = 23.44 * math.pi / 180

    return G, MY, KS_TOLERANCE, GRAV_CONST, EARTH_RADIUS, OBLIQUITY

def get_SRP_globals():
    sigma_star = 1.53 * 10**(-3)  # kg / m^2
    L_s = 3.8205*10**26  # W 3.8275*10**26
    R_s = 696340000  # m
    c = 299792458  # m/s
    return sigma_star, L_s, R_s, c

def get_normalization_factors():
    return {"SMA": 300000000, "ECC": 1, "INC": 1, "t_prop": 100*24*3600, "time_offset": 30*3600}