import numpy as np

def check_conditions(sc, origin, normal, sample_points: list, fill_num=0, return_type=None, cutoff_dist=None):

    def condition_1(sample_points):
        # Check input
        if len(sample_points) > len(sc.integration_points):
            print("Warning! More sample points requested than where integrated! Assuming all available sample points!")
            sample_points = range(len(sc.integration_points))

        traj = sc.trajectory_track
        is_feasible = False

        resample_dict = dict()
        dist_dict = dict()

        for i in sample_points:
            s_vec = np.array([traj[0][i], traj[1][i], traj[2][i], traj[3][i], traj[4][i], traj[5][i]])
            if planar_boundary(x=s_vec, origin=origin, normal=normal):
                is_feasible = True
                if return_type is not None:
                    list_of_svec, dist = fill_set(origin=origin, normal=normal, x=s_vec, fill_num=fill_num, return_type=return_type, cutoff_dist=cutoff_dist)
                    if sc.integration_points[i] not in resample_dict.keys():
                        resample_dict[sc.integration_points[i]] = list_of_svec
                        dist_dict[sc.integration_points[i]] = dist
                    else:
                        resample_dict[sc.integration_points[i]] += list_of_svec
                        dist_dict[sc.integration_points[i]] =+ dist

        return is_feasible, resample_dict, dist_dict


    return condition_1(sample_points)

def planar_boundary(origin, normal, x):
    # Inputs as numpy arrays
    # Defines a plane and direction. Checks, is spacecraft is on valid side of plane. --> normal-vector points towards the fasible side

    projection = (np.dot((x-origin), normal) / np.dot(normal, normal)) * normal

    for i in range(6):
        if (projection[i] != 0 and normal[i] != 0) and (projection[i] * normal[i]) < 0:
            return False
    return True


def fill_set(origin, normal, x, fill_num, return_type: str, cutoff_dist=None):

    if fill_num < 1:
        raise ValueError("Fill_num has to be equal to or greater than 1!")

    # Creates additional sample points to the feasibility plane
    projection = (np.dot((x - origin), normal) / np.dot(normal, normal)) * normal

    # Check for minimum 6dof distance:
    if (cutoff_dist is not None) and (np.linalg.norm(projection) > cutoff_dist):
        return [], []
    base = x - projection

    point_arrays = [[] for k in range(6)]
    # state_vec_array = [[] for k in range(fill_num)]

    for i in range(6):
        point_arrays[i] = np.linspace(base[i], x[i], fill_num).tolist()

    # Convert to state vectors
    if return_type == "new":
        state_vec_array = [[] for _ in range(fill_num - 1)]
        dist_array = [_ for _ in range(fill_num - 1)]
        for k in range(fill_num-1):
            state_vec_array[k] = [point_arrays[i][k] for i in range(6)]
            dist_array[k] = np.linalg.norm((np.dot((np.array(state_vec_array[k]) - origin), normal) / np.dot(normal, normal)) * normal)

    elif return_type == "base":
        state_vec_array =  [[point_arrays[i][0] for i in range(6)]]
        # dist_array = [np.linalg.norm((np.dot((np.array(state_vec_array[0]) - origin), normal) / np.dot(normal, normal)) * normal)]
        dist_array = [0]

    elif return_type == "all":
        state_vec_array = [[] for _ in range(fill_num)]
        dist_array = [_ for _ in range(fill_num)]
        for k in range(fill_num):
            state_vec_array[k] = [point_arrays[i][k] for i in range(6)]
            dist_array[k] = np.linalg.norm((np.dot((np.array(state_vec_array[k]) - origin), normal) / np.dot(normal, normal)) * normal)

    elif return_type == "on_set":
        state_vec_array = [[point_arrays[i][-1] for i in range(6)]]
        dist_array = [np.linalg.norm((np.dot((np.array(state_vec_array[-1]) - origin), normal) / np.dot(normal, normal)) * normal)]
    else:
        raise ValueError("Wrong return type! Use 'new', 'base' or 'all'")

    return state_vec_array, dist_array

if __name__ == "__main__":
    normal = np.array([1, 1, 0, 0, 0, 0])
    origin = np.array([2, 1, 0, 0, 0, 0])
    x = np.array([5, 0, 0, 0, 0, 0])
    state_vecs = fill_set(origin, normal, x, 2, "base")
    pass

