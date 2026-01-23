from src.globals import colors

color_data, size_data = colors.get_color_data()


def spacecraft_slice_group(list_of_spacecraft, index=0, state_type=None):
    # Filter spacecraft into lists edge, center, generic and feasible
    if state_type is None:
        state_type = ["x", "y", "z", "vx", "vy", "vz"]
    edge = {}
    center = {}
    generic = {}
    feasible = {}
    resampled = {}

    for state in state_type:
        edge[state] = []
        center[state] = []
        generic[state] = []
        feasible[state] = []
        resampled[state] = []

    # Sort the trajectory into the lists based on spacecraft type
    for sc in list_of_spacecraft:
        if sc.is_feasible:
            for k, key in enumerate(feasible.keys()):
                feasible[key].append(sc.trajectory_track[k][index])

        elif sc.is_center_spacecraft:
            for k, key in enumerate(center.keys()):
                center[key].append(sc.trajectory_track[k][index])

        elif sc.is_edge_spacecraft:
            for k, key in enumerate(edge.keys()):
                edge[key].append(sc.trajectory_track[k][index])

        else:
            # Is a generic spacecraft
            for k, key in enumerate(generic.keys()):
                generic[key].append(sc.trajectory_track[k][index])

    list_of_dict = [edge, center, generic, feasible]
    list_of_color = [color_data["edge"], color_data["center"], color_data["generic"], color_data["feasible"]]
    list_of_alpha = [1, 1, 1, 1]

    return list_of_dict, list_of_color, list_of_alpha


def full_trajectory_group(list_of_spacecraft, state_type=None):
    if state_type is None:
        state_type = ["x", "y", "z", "vx", "vy", "vz"]

    edge = {}
    center = {}
    generic = {}
    feasible = {}

    for state in state_type:
        edge[state] = []
        center[state] = []
        generic[state] = []
        feasible[state] = []

    # Attach a timescale
    edge["t"] = []
    center["t"] = []
    generic["t"] = []
    feasible["t"] = []

    # Sort the trajectory into the lists based on spacecraft type
    for sc in list_of_spacecraft:
        if sc.is_feasible:
            for k, key in enumerate(state_type):
                feasible[key].append(sc.trajectory_track[k])
            feasible["t"].append(sc.integration_points)

        elif sc.is_center_spacecraft:
            for k, key in enumerate(state_type):
                center[key].append(sc.trajectory_track[k])
            center["t"].append(sc.integration_points)

        elif sc.is_edge_spacecraft:
            for k, key in enumerate(state_type):
                edge[key].append(sc.trajectory_track[k])
            edge["t"].append(sc.integration_points)

        else:
            # Is a generic spacecraft
            for k, key in enumerate(state_type):
                generic[key].append(sc.trajectory_track[k])
            generic["t"].append(sc.integration_points)

    list_of_dict = [edge, center, generic, feasible]
    list_of_color = [color_data["edge"], color_data["center"], color_data["generic"], color_data["feasible"]]
    list_of_alpha = [1, 1, 1, 1]

    return list_of_dict, list_of_color, list_of_alpha


def full_kepler_parameter_group(list_of_spacecraft, param_type=None):
    if param_type is None:
        param_type = ["SMA", "ECC", "INC", "RAAN", "APERI", "TAEPO"]

    edge = {}
    center = {}
    generic = {}
    feasible = {}

    for state in param_type:
        edge[state] = []
        center[state] = []
        generic[state] = []
        feasible[state] = []

    # Attach a timescale
    edge["t"] = []
    center["t"] = []
    generic["t"] = []
    feasible["t"] = []

    # Sort the trajectory into the lists based on spacecraft type
    for sc in list_of_spacecraft:
        if sc.is_feasible:
            for k, key in enumerate(param_type):
                feasible[key].append(sc.orbital_parameters_track[k])
            feasible["t"].append(sc.integration_points)

        elif sc.is_center_spacecraft:
            for k, key in enumerate(param_type):
                center[key].append(sc.orbital_parameters_track[k])
            center["t"].append(sc.integration_points)

        elif sc.is_edge_spacecraft:
            for k, key in enumerate(param_type):
                edge[key].append(sc.orbital_parameters_track[k])
            edge["t"].append(sc.integration_points)

        else:
            # Is a generic spacecraft
            for k, key in enumerate(param_type):
                generic[key].append(sc.orbital_parameters_track[k])
            generic["t"].append(sc.integration_points)

    list_of_dict = [edge, center, generic, feasible]
    list_of_color = [color_data["edge"], color_data["center"], color_data["generic"], color_data["feasible"]]
    list_of_alpha = [1, 1, 1, 1]

    return list_of_dict, list_of_color, list_of_alpha


def full_scalar_arrays_group(list_of_spacecraft, param_type=None):
    if param_type is None:
        param_type = ["C3", "vel", "pos"]

    # Simple parameter check:
    for par in param_type:
        if par not in ["C3", "vel", "pos"]:
            raise ValueError("Wrong parameter! Use C3, vel or pos!")

    edge = {}
    center = {}
    generic = {}
    feasible = {}

    for state in param_type:
        edge[state] = []
        center[state] = []
        generic[state] = []
        feasible[state] = []

    # Attach a timescale
    edge["t"] = []
    center["t"] = []
    generic["t"] = []
    feasible["t"] = []

    # Sort the trajectory into the lists based on spacecraft type
    for sc in list_of_spacecraft:
        if sc.is_feasible:
            for k, key in enumerate(param_type):
                if key == "C3":
                    feasible[key].append(sc.C3_track)
                elif key == "vel":
                    feasible[key].append(sc.vel_mag_track)
                elif key == "pos":
                    feasible[key].append(sc.slant_range_track)

            feasible["t"].append(sc.integration_points)

        elif sc.is_center_spacecraft:
            for k, key in enumerate(param_type):
                if key == "C3":
                    center[key].append(sc.C3_track)
                elif key == "vel":
                    center[key].append(sc.vel_mag_track)
                elif key == "pos":
                    center[key].append(sc.slant_range_track)

            center["t"].append(sc.integration_points)

        elif sc.is_edge_spacecraft:
            for k, key in enumerate(param_type):
                if key == "C3":
                    edge[key].append(sc.C3_track)
                elif key == "vel":
                    edge[key].append(sc.vel_mag_track)
                elif key == "pos":
                    edge[key].append(sc.slant_range_track)

            edge["t"].append(sc.integration_points)
            # edge["t"].append(sc.integration_points)

        else:
            # Is a generic spacecraft
            for k, key in enumerate(param_type):
                if key == "C3":
                    generic[key].append(sc.C3_track)
                elif key == "vel":
                    generic[key].append(sc.vel_mag_track)
                elif key == "pos":
                    generic[key].append(sc.slant_range_track)

            generic["t"].append(sc.integration_points)

    list_of_dict = [edge, center, generic, feasible]
    list_of_color = [color_data["edge"], color_data["center"], color_data["generic"], color_data["feasible"]]
    list_of_alpha = [1, 1, 1, 1]

    return list_of_dict, list_of_color, list_of_alpha


def full_body_dist_groups(list_of_spacecraft):
    edge = {}
    center = {}
    generic = {}
    feasible = {}

    body_list = list(list_of_spacecraft[0].body_distances.keys())

    for body in body_list:
        edge[body] = []
        center[body] = []
        generic[body] = []
        feasible[body] = []

    # Attach a timescale
    edge["t"] = []
    center["t"] = []
    generic["t"] = []
    feasible["t"] = []

    # Sort the trajectory into the lists based on spacecraft type
    for sc in list_of_spacecraft:

        if sc.is_feasible:
            for b, body in enumerate(body_list):
                feasible[body].append(sc.body_distances[body])

            feasible["t"].append(sc.integration_points)

        elif sc.is_center_spacecraft:
            for b, body in enumerate(body_list):
                center[body].append(sc.body_distances[body])

            center["t"].append(sc.integration_points)

        elif sc.is_edge_spacecraft:
            for b, body in enumerate(body_list):
                edge[body].append(sc.body_distances[body])

            edge["t"].append(sc.integration_points)

        else:
            # Is a generic spacecraft
            for b, body in enumerate(body_list):
                generic[body].append(sc.body_distances[body])

            generic["t"].append(sc.integration_points)

    list_of_dict = [edge, center, generic, feasible]
    list_of_color = [color_data["edge"], color_data["center"], color_data["generic"], color_data["feasible"]]
    list_of_alpha = [1, 1, 1, 1]

    return list_of_dict, list_of_color, list_of_alpha


def resampled_slice_group(list_of_resample_sc, index=0, state_type=None):
    # Filter spacecraft into lists edge, center, generic and feasible
    if state_type is None:
        state_type = ["x", "y", "z", "vx", "vy", "vz"]

    resample = dict()

    for state in state_type:
        resample[state] = []

    # Sort the trajectory into the lists based on spacecraft type
    for sc in list_of_resample_sc:
        for k, key in enumerate(resample.keys()):
            resample[key].append(sc.trajectory_track[k][index])

    list_of_dict = [resample]
    list_of_color = [color_data["resampled"]]
    list_of_alpha = [1]

    return list_of_dict, list_of_color, list_of_alpha
