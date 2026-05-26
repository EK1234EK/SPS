class Sail_parameters:
    def __init__(self):
        self.sets = {"ACS3":
            {
                "sigma": 5,
                "r_f": 0.9,
                "r_b": 0.9,
                "s_f": 0.82,
                "s_b": 0.82,
                "B_f": 0.79,
                "B_b": 0.67,
                "e_f": 0.03,
                "e_b": 0.6
            }
        }

    def fetch(self, name):
        return self.sets[name]
