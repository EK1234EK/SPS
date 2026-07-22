class Sail_parameters:
    def __init__(self, sigma=0.02):
        self.sets = {"ACS3":
            {
                "sigma": sigma,
                "r_f": 0.9,
                "r_b": 0.9,
                "s_f": 0.82,
                "s_b": 0.82,
                "B_f": 0.79,
                "B_b": 0.67,
                "e_f": 0.03,
                "e_b": 0.6
            },
                    "NEA_scout":
            {
                "sigma": 1,
                "r_f": 0.91,
                "r_b": 0.91,
                "s_f": 0.89,
                "s_b": 0.89,
                "B_f": 0.79,
                "B_b": 0.67,
                "e_f": 0.025,
                "e_b": 0.27
            },
                    "ideal":
            {
                "sigma": sigma,
                "r_f": None,
                "r_b": None,
                "s_f": None,
                "s_b": None,
                "B_f": None,
                "B_b": None,
                "e_f": None,
                "e_b": None
            },
                    "ideal_real":
            {
                "sigma": sigma,
                "r_f": 0.95,
                "r_b": 0.95,
                "s_f": 1,
                "s_b": 1,
                "B_f": 2/3,
                "B_b": 2/3,
                "e_f": 0,
                "e_b": 0
            },
                    "Heligyro":
            {
                "sigma": sigma,
                "r_f": 0.88,
                "r_b": 0.88,
                "s_f": 0.94,
                "s_b": 0.94,
                "B_f": 0.79,
                "B_b": 0.55,
                "e_f": 0.05,
                "e_b": 0.55
            }
        }

    def fetch(self, name):
        return self.sets[name]
