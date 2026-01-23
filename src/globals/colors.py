def get_color_data():
    colors = dict()

    colors["generic"] = [0, 0, 1]
    colors["edge"] = [0, 1, 0]
    colors["center"] = [0.8, 0.2, 1]
    colors["feasible"] = [0.2, 1, 1]
    colors["resampled"] = [1, 0, 0]

    colors["planet_traj"] = [1, 1, 1]
    colors["special_tail_color"] = [0.5, 0.5, 0.5]
    colors["tail_line_width"] = 0.5

    colors["background"] = (0, 0, 0)  # (21/265, 20/265, 27/265)
    colors["ticks"] = (1, 1, 1)
    colors["grid"] = (0.7, 0.6, 1)
    colors["map"] = "jet"

    colors["lagrange_marker_symbol"] = "D"
    colors["lagrange_marker_color"] = [0.5, 1, 0.5]

    size = dict()
    size["sc"] = 5
    size["dia_linewidth"] = 0.5
    size["plot_alpha"] = 1
    size["lagrange_marker_size"] = 10

    return colors, size
