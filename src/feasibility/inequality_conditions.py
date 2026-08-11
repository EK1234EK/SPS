import numpy as np


def check_and(sc):
    def condition_1(sc):
        if min(sc.body_distances["Moon"]) < 3e7:
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
        if 300097940.4568403 < sc.slant_range_track[-1] < 340497940.4568403:
            return True

    def condition_6(sc):
        if max(sc.trajectory_track[0]) < 1 and min(sc.trajectory_track[0]) > 0.6:
            return True

    def condition_7(sc):
        if abs(((sc.trajectory_track[0][-1] - 0.8346191402770146)**2 + sc.trajectory_track[1][-1]**2 + sc.trajectory_track[2][-1]**2)**0.5) < 0.01 and abs((sc.trajectory_track[3][-1] ** 2 + sc.trajectory_track[4][-1] ** 2 + sc.trajectory_track[5][-1] ** 2) ** 0.5) < 0.01:
            return True
    return condition_5(sc)
