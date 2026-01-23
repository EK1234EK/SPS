import numpy as np


def check_and(sc):
    def condition_1(sc):
        if min(sc.body_distances["Jupiter"]) < 1.355e9:
            return True

    def condition_2(sc):
        if min(sc.body_distances["Saturn"]) < 1e9:
            return True

    def condition_3(sc):
        sc.condition_value = [sc.C3_track[-1]]
        if sc.C3_track[-1] > -0.75e9:
            return True

    def condition_4(sc):
        sc.condition_value = [sc.trajectory_track[0][0]]
        if sc.trajectory_track[0][0] < 0:
            return True

    def condition_5(sc):
        if min(sc.body_distances["body_1"]) < 0.9e9:
            return True

    def condition_6(sc):
        if (sc.trajectory_track[0][-1] ** 2 + sc.trajectory_track[1][-1] ** 2 + sc.trajectory_track[2][-1] ** 2)**0.5 > 0.9:
            return True

    return condition_6(sc)
