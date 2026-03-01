import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.stats import alpha

from src.analysis import plots_additional
from src.globals import colors
import matplotx

color_data, size_data = colors.get_color_data()
mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'


class graph_output:
    def __init__(self, list_of_spacecraft, list_of_resampled_spacecraft, list_of_special_spacecraft, force_model,
                 animated=True, frame_delay=1, axis_visibility=True, fps=None, init_azim=0, init_elevation=90):
        self.list_of_spacecraft = list_of_spacecraft
        self.list_of_resampled_spacecraft = list_of_resampled_spacecraft
        self.lst_spec_sc = list_of_special_spacecraft
        self.integration_points = []
        self.bodies_traj = None
        self.bodies_plot_trajectories = None
        self.animated = animated
        self.frame_delay = frame_delay
        self.frames = 200
        self.frame_multiplier = 1
        self.force_model = force_model
        self.sensitivity = None

        self.axis_visibility = axis_visibility
        self.fps = fps

        self.slices = [["x", "y"], ["x", "z"], ["y", "z"], ["vx", "vy"], ["vx", "vz"], ["vy", "vz"]]
        self.slice_index = None

        self.figure_counter = 0

        self.get_reference_data()

        # plt.style.use(matplotx.styles.aura["dark-soft"]) dark_background
        plt.style.use('dark_background')
        plt.rc('axes', edgecolor=(1, 1, 1))

    def get_reference_data(self):
        if self.list_of_spacecraft:
            self.integration_points = self.list_of_spacecraft[0].integration_points
            self.bodies_traj = self.force_model.propagate_body_states(self.list_of_spacecraft[0].integration_points)
        elif self.lst_spec_sc:
            self.integration_points = self.lst_spec_sc[0].integration_points
            self.bodies_traj = self.force_model.propagate_body_states(self.lst_spec_sc[0].integration_points)
        else:
            raise ValueError("No integration points found!")

        self.bodies_plot_trajectories = self.force_model.get_plotting_track()
        pass

    def moving_map_plot(self, plot_central_attractor=True,
                        match_tail_color=False,
                        plot_potential=False,
                        plot_planet_endpoint=True,
                        init_azim=0,
                        init_elevation=90,
                        azim_rate=0,
                        elevation_rate=0,
                        k_modulo=None):

        fig = plt.figure(self.figure_counter + 1, figsize=(16, 9))
        self.figure_counter += 1
        ax = fig.add_subplot(111, projection='3d')
        ax.set_position([0.0, 0.0, 1.0, 0.95])

        fig.patch.set_facecolor(color_data["background"])

        ax.view_init(elev=init_elevation, azim=init_azim, roll=0)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        ax.xaxis.label.set_color(color_data["ticks"])
        ax.yaxis.label.set_color(color_data["ticks"])
        ax.zaxis.label.set_color(color_data["ticks"])

        ax.tick_params(axis='x', colors=color_data["ticks"])
        ax.tick_params(axis='y', colors=color_data["ticks"])
        ax.tick_params(axis='z', colors=color_data["ticks"])

        ax.spines['left'].set_color(color_data["ticks"])
        ax.spines['top'].set_color(color_data["ticks"])
        ax.spines['right'].set_color(color_data["ticks"])
        ax.spines['bottom'].set_color(color_data["ticks"])

        n_samp = len(self.lst_spec_sc)
        cmap = plt.colormaps[color_data["map"]]
        colors = cmap(np.linspace(0, 1, n_samp))

        # Create the initial plots
        artists = []
        list_of_dict, list_of_color, list_of_alpha = plots_additional.spacecraft_slice_group(self.list_of_spacecraft,
                                                                                             index=0)

        if self.list_of_resampled_spacecraft:
            list_of_dict_res, list_of_color_res, list_of_alpha_res = plots_additional.resampled_slice_group(
                self.list_of_resampled_spacecraft, index=0)
            list_of_dict += list_of_dict_res
            list_of_color += list_of_color_res
            list_of_alpha += list_of_alpha_res

        # Get the valid groups (The ones that actually have some spacecraft associated with them)
        valid_groups = []
        valid_groups_key = []
        for k, lst in enumerate(list_of_dict):
            if lst["x"] != []:
                valid_groups.append(k)
                valid_groups_key.append(lst)

        special_artists = []
        special_tails = []
        for i, sc_spec in enumerate(self.lst_spec_sc):
            color = sc_spec.plot_color
            name = sc_spec.display_name
            if isinstance(color, str):
                spec_artist = ax.scatter([], [], [], color=colors[i], label=name, s=size_data["sc"])
            else:
                spec_artist = ax.scatter([], [], [], color=color, label=name, s=size_data["sc"])
            special_artists.append(spec_artist)

            if match_tail_color:
                if isinstance(sc_spec.plot_color, str):
                    spec_tail = ax.plot([], [], [], color=colors[i],
                                        linewidth=color_data["tail_line_width"])
                else:
                    spec_tail = ax.plot([], [], [], color=sc_spec.plot_color,
                                        linewidth=color_data["tail_line_width"])
                special_tails.append(spec_tail)
            else:
                spec_tail = ax.plot([], [], [], color=color_data["special_tail_color"],
                                    linewidth=color_data["tail_line_width"])
                special_tails.append(spec_tail)

        group_artists = []
        for i in valid_groups:
            color = list_of_color[i]
            group = list_of_dict[i]

            point_group = ax.scatter(group["x"][0], group["y"][0], group["z"][0], color=color, s=size_data["sc"])
            group_artists.append(point_group)

        body_artists = []
        for _ in self.bodies_plot_trajectories.keys():
            body_end = ax.scatter([], [], [], color=color_data["planet_traj"], s=40)
            body_artists.append(body_end)

        artists.append([group_artists, body_artists, special_artists, special_tails])

        # Begin of init()
        def init():
            max_b = [0, 0, 0]
            min_b = [0, 0, 0]

            max_sc = [0, 0, 0]
            min_sc = [0, 0, 0]

            max_ax = [0, 0, 0]
            min_ax = [0, 0, 0]

            diffs = [0, 0, 0]
            ax_upper = [0, 0, 0]
            ax_lower = [0, 0, 0]

            for body in self.bodies_traj.keys():
                for k in range(3):
                    if max(self.bodies_plot_trajectories[body][k]) > max_b[k]: max_b[k] = max(
                        self.bodies_plot_trajectories[body][k])
                    if min(self.bodies_plot_trajectories[body][k]) < min_b[k]: min_b[k] = min(
                        self.bodies_plot_trajectories[body][k])

            for spacecraft in self.list_of_spacecraft + self.list_of_resampled_spacecraft + self.lst_spec_sc:
                traj = spacecraft.trajectory_track
                for k in range(3):
                    if max(traj[k]) > max_sc[k]: max_sc[k] = max(traj[k])
                    if min(traj[k]) < min_sc[k]: min_sc[k] = min(traj[k])

            # Get the global min and max values
            for k in range(3):
                max_ax[k] = max([max_b[k], max_sc[k]])
                min_ax[k] = min([min_b[k], min_sc[k]])
                diffs[k] = max_ax[k] - min_ax[k]

            # Get the scale factors for each dimension such that
            max_diff = max([max_ax[k] - min_ax[k] for k in range(3)])

            for k in range(3):
                # scale_factors[k] = max_diff / (max_ax[k] - min_ax[k])
                if max_ax[k] == min_ax[k]:
                    ax_upper[k] = 1
                    ax_lower[k] = -1
                else:
                    """ax_upper[k] = max_ax[k] * (max_diff / (max_ax[k] - min_ax[k]))
                    ax_lower[k] = min_ax[k] * (max_diff / (max_ax[k] - min_ax[k]))

                    ax_upper[k] += 0.2 * abs(ax_upper[k])
                    ax_lower[k] -= 0.2 * abs(ax_lower[k])"""
                    overhead = max_diff - (max_ax[k] - min_ax[k])
                    ax_upper[k] = max_ax[k] + overhead * 0.6
                    ax_lower[k] = min_ax[k] - overhead * 0.6

            # Plot the body trajectories
            for body in self.bodies_plot_trajectories.keys():
                b1_x1 = self.bodies_plot_trajectories[body][0]
                b1_x2 = self.bodies_plot_trajectories[body][1]
                b1_x3 = self.bodies_plot_trajectories[body][2]
                ax.plot(b1_x1, b1_x2, b1_x3, color=color_data["planet_traj"], linewidth=1, linestyle=(0, (5, 5)))

            # Plot a blob at the central attractor position
            if plot_central_attractor:
                ax.scatter([self.force_model.central_attractor_pos[0]],
                           [self.force_model.central_attractor_pos[1]],
                           [self.force_model.central_attractor_pos[2]],
                           color=[1, 0.7, 0.7], s=50)

            """# Plot the CR3BP pseudo-potential, if corresponding force model
            if self.force_model.is_CR3BP and plot_potential:
                xv, yv, potential_map = self.force_model.get_potential_field(resolution=1000, lim_x=[ax_lower[0], ax_upper[0]], lim_y=[ax_lower[1], ax_upper[1]])
                ax.contour(xv, yv, potential_map, 500, lw=1, cmap="autumn_r", extend3d=False, offset=0)"""

            ax.set_xlim(ax_lower[0], ax_upper[0])
            ax.set_ylim(ax_lower[1], ax_upper[1])
            ax.set_zlim(ax_lower[2], ax_upper[2])

            fig.set_facecolor(color_data["background"])
            ax.set_facecolor(color_data["background"])
            ax.grid(False)
            ax.xaxis.set_pane_color(color_data["background"])
            ax.yaxis.set_pane_color(color_data["background"])
            ax.zaxis.set_pane_color(color_data["background"])

            if not self.axis_visibility:
                ax.set_axis_off()
            # fig.tight_layout()

        # In case of CR3BP force model, plot the Lagrange points as well:
        if self.force_model.is_CR3BP:
            self.force_model.get_lagrange_points()
            for key in self.force_model.lagrange_points.keys():
                ax.scatter(self.force_model.lagrange_points[key][0],
                           self.force_model.lagrange_points[key][1],
                           self.force_model.lagrange_points[key][2],
                           marker=color_data["lagrange_marker_symbol"],
                           s=size_data["lagrange_marker_size"],
                           color=color_data["lagrange_marker_color"])

        # End of init()
        ####################
        # Begin of animate(k)
        def animate(k_in):
            k = math.floor(k_in * self.frame_multiplier)  # Account for the case of no animation - just show the final state

            if k_modulo and self.animated:
                k = math.floor(k * k_modulo)

            list_of_dict, list_of_color, list_of_alpha = plots_additional.spacecraft_slice_group(
                self.list_of_spacecraft, index=k)

            if self.list_of_resampled_spacecraft:
                list_of_dict_res, list_of_color_res, list_of_alpha_res = plots_additional.resampled_slice_group(
                    self.list_of_resampled_spacecraft, index=k)
                list_of_dict += list_of_dict_res
                list_of_color += list_of_color_res
                list_of_alpha += list_of_alpha_res

            for idx, g in enumerate(valid_groups):
                endpoint = artists[0][0][idx]
                this_group = list_of_dict[g]
                # trajectory = spacecraft.trajectory_track
                endpoint._offsets3d = (this_group["x"], this_group["y"], this_group["z"])
                """if plot_tail:
                    trajectory_track = artists_list[0]
                    trajectory_track.set_data_3d(trajectory[0][0:k + 1], trajectory[1][0:k + 1], trajectory[2][0:k + 1])"""

            for i_spec, spec_sc in enumerate(self.lst_spec_sc):
                trajectory_track = artists[0][2][i_spec]
                tail = artists[0][3][i_spec][0]
                trajectory_track._offsets3d = (
                    [spec_sc.trajectory_track[0][k]], [spec_sc.trajectory_track[1][k]],
                    [spec_sc.trajectory_track[2][k]])
                tail.set_data_3d(spec_sc.trajectory_track[0][0:k + 1], spec_sc.trajectory_track[1][0:k + 1],
                                 spec_sc.trajectory_track[2][0:k + 1])

            # Plotting the body endpoints
            body_artists_iter = artists[0][1]
            if plot_planet_endpoint:
                for idx, body in enumerate(self.bodies_traj.keys()):
                    b1_x1 = self.bodies_traj[body][0][k]
                    b1_x2 = self.bodies_traj[body][1][k]
                    b1_x3 = self.bodies_traj[body][2][k]
                    # artists_list[idx + 2]._offsets3d = ([b1_x1], [b1_x2], [b1_x3])
                    body_artists_iter[idx]._offsets3d = ([b1_x1], [b1_x2], [b1_x3])
                    # print(([b1_x1], [b1_x2], [b1_x3]))

            if self.integration_points[k] < 86000 / 2:
                ax.set_title('Elapsed time: ' + str(round(self.integration_points[k], 2)) + ' / ' + str(
                    round(self.integration_points[-1], 2)) + ' s')
            else:
                ax.set_title('Elapsed time: ' + str(round(self.integration_points[k] / (24 * 3600), 2)) + ' / ' + str(
                    round(self.integration_points[-1] / (24 * 3600), 2)) + ' d')

            #Rotate
            if azim_rate != 0 or elevation_rate != 0:
                ax.view_init(azim=init_azim+k*azim_rate, elev=init_elevation+k*elevation_rate)

            return artists

        # End of animate(k)
        ###################
        # Call the animation loop

        frames = len(self.integration_points)

        if not self.animated:
            frames = 2
            self.frame_multiplier = (len(self.integration_points) - 1)

        else:
            if k_modulo:
                frames = math.floor(frames / k_modulo)

        anim = FuncAnimation(fig, animate, frames=frames, interval=self.frame_delay, repeat=False, init_func=init(),
                             blit=False)

        if self.fps is not None:
            anim.save(filename="animation_out/1.gif", fps=self.fps, dpi=200)
        if not self.axis_visibility:
            plt.legend()

        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()

        plt.show()

    def state_space_slice(self, index=0, slices=[["x", "y"]], edge=True, center=True, generic=True, feasible=True,
                          resample=True):
        fig = plt.figure(self.figure_counter + 1, figsize=(5 * 2, 3.5 * 5 / 3))
        self.figure_counter += 1
        fig.set_facecolor(color_data["background"])

        groups_to_plot = [edge, center, generic, feasible, resample]
        list_of_dict, list_of_color, list_of_alpha = plots_additional.spacecraft_slice_group(
            self.list_of_spacecraft,
            index=index)

        if resample:
            list_of_dict_res, list_of_color_res, list_of_alpha_res = plots_additional.resampled_slice_group(
                self.list_of_resampled_spacecraft, index=index)
            list_of_dict += list_of_dict_res
            list_of_color += list_of_color_res
            list_of_alpha += list_of_alpha_res

        # The order of the spacecraft categories is: edge, center, generic, feasible, resample

        n_samp = len(self.list_of_spacecraft)
        cmap = plt.colormaps[color_data["map"]]
        colors = cmap(np.linspace(0, 1, n_samp))

        patches = ["Boundary", "Center", "Generic", "Feasible", "Resampled"]

        # Getting the arrangement of the subplots based on the number of bodies:
        if len(slices) == 1:
            height = 1
            width = 1
        elif len(slices) == 2:
            height = 1
            width = 2
        elif len(slices) > 2:
            height = 2
            width = math.ceil(len(slices) / 2)
        else:
            raise ValueError("Length of the list of slices is less than one! It is " + str(len(slices)))

        axes = dict()
        # Generate the axes
        for k, ind_slice in enumerate(slices):
            if len(ind_slice) == 2:
                axis = fig.add_subplot(height, width, k + 1)
                if self.integration_points[index] < 86000 / 2:
                    axis.set_title(ind_slice[0] + " - " + ind_slice[1] + "    " + str(
                        round(self.integration_points[index], 2)) + " / " + str(
                        round(self.integration_points[-1], 2)) + " [s]")
                else:
                    axis.set_title(ind_slice[0] + " - " + ind_slice[1] + "    " + str(
                        round(self.integration_points[index] / (24 * 3600), 2)) + " / " + str(
                        round(self.integration_points[-1] / (24 * 3600), 2)) + " [d]")

                axis.xaxis.label.set_color(color_data["ticks"])
                axis.yaxis.label.set_color(color_data["ticks"])

                axis.tick_params(axis='x', colors=color_data["ticks"])
                axis.tick_params(axis='y', colors=color_data["ticks"])

                axis.spines['left'].set_color(color_data["ticks"])
                axis.spines['top'].set_color(color_data["ticks"])
                axis.spines['right'].set_color(color_data["ticks"])
                axis.spines['bottom'].set_color(color_data["ticks"])

                axis.grid(visible=True, color=color_data["grid"])

                axes[tuple(ind_slice)] = axis
            elif len(ind_slice) == 3:
                axis = fig.add_subplot(height, width, k + 1, projection='3d')

                if self.integration_points[index] < 86000 / 2:
                    axis.set_title(ind_slice[0] + " - " + ind_slice[1] + " - " + ind_slice[2] + "    " + str(
                    round(self.integration_points[index], 2)) + " / " + str(
                    round(self.integration_points[-1], 2)) + " [s]")
                else:
                    axis.set_title(ind_slice[0] + " - " + ind_slice[1] + " - " + ind_slice[2] + "    " + str(
                    round(self.integration_points[index] / (24 * 3600), 2)) + " / " + str(
                    round(self.integration_points[-1] / (24 * 3600), 2)) + " [d]")

                axis.xaxis.pane.fill = False
                axis.yaxis.pane.fill = False
                axis.zaxis.pane.fill = False

                # Now set color to white (or whatever is "invisible")
                axis.xaxis.label.set_color(color_data["ticks"])
                axis.yaxis.label.set_color(color_data["ticks"])
                axis.zaxis.label.set_color(color_data["ticks"])

                axis.tick_params(axis='x', colors=color_data["ticks"])
                axis.tick_params(axis='y', colors=color_data["ticks"])
                axis.tick_params(axis='z', colors=color_data["ticks"])

                axis.spines['left'].set_color(color_data["ticks"])
                axis.spines['top'].set_color(color_data["ticks"])
                axis.spines['right'].set_color(color_data["ticks"])
                axis.spines['bottom'].set_color(color_data["ticks"])

                axis.grid(visible=True, color=color_data["grid"])

                axes[tuple(ind_slice)] = axis
            else:
                raise ValueError("Wrong sequence in slice array! Use length 2 or 3!")

            axis.xaxis.label.set_color(color_data["ticks"])
            axis.yaxis.label.set_color(color_data["ticks"])

            axis.tick_params(axis='x', colors=color_data["ticks"])
            axis.tick_params(axis='y', colors=color_data["ticks"])

            axis.spines['left'].set_color(color_data["ticks"])
            axis.spines['top'].set_color(color_data["ticks"])
            axis.spines['right'].set_color(color_data["ticks"])
            axis.spines['bottom'].set_color(color_data["ticks"])

        # Plot the data to the axes and gernerate colors
        n_samp = len(self.lst_spec_sc)
        cmap = plt.colormaps[color_data["map"]]
        colors = cmap(np.linspace(0, 1, n_samp))

        for ind_slice in slices:
            if len(ind_slice) == 2:
                # Plot the standard group data
                for i, group in enumerate(list_of_dict):
                    if groups_to_plot[i] and group[ind_slice[0]] != []:
                        axes[tuple(ind_slice)].scatter(group[ind_slice[0]],
                                                       group[ind_slice[1]],
                                                       color=list_of_color[i],
                                                       s=size_data["sc"],
                                                       label=patches[i]
                                                       )

                # Plot the data of the special spacecraft, if there are any
                # First, get the indeces of the dimensions to plot

                idx_1 = ["x", "y", "z", "vx", "vy", "vz"].index(ind_slice[0])
                idx_2 = ["x", "y", "z", "vx", "vy", "vz"].index(ind_slice[1])
                for i, sc in enumerate(self.lst_spec_sc):
                    axes[tuple(ind_slice)].scatter(sc.trajectory_track[idx_1][index],
                                                   sc.trajectory_track[idx_2][index],
                                                   color=colors[i] if sc.plot_color == "map" else sc.plot_color,
                                                   s=size_data["sc"],
                                                   label=sc.display_name
                                                   )

                # Get the units
                if ind_slice[0] in ["x", "y", "z"]:
                    xl = ind_slice[0] + " [m]"
                else:
                    xl = ind_slice[0] + " [m/s]"

                if ind_slice[1] in ["x", "y", "z"]:
                    yl = ind_slice[1] + " [m]"
                else:
                    yl = ind_slice[1] + " [m/s]"
                axes[tuple(ind_slice)].set_xlabel(xl)
                axes[tuple(ind_slice)].set_ylabel(yl)
                axes[tuple(ind_slice)].grid(visible=True, which="both")

                axes[tuple(ind_slice)].set_facecolor(color_data["background"])

                #fig.tight_layout()
                lgnd = axes[tuple(ind_slice)].legend(loc='upper left')
                lgnd.set_draggable(True)
                for handle in lgnd.legend_handles:
                    handle.set_sizes([50])
                # axes[tuple(ind_slice)].set_facecolor('xkcd:midnight')
            else:
                for i, group in enumerate(list_of_dict):
                    if groups_to_plot[i] and group[ind_slice[0]] != []:
                        axes[tuple(ind_slice)].scatter(group[ind_slice[0]],
                                                       group[ind_slice[1]],
                                                       group[ind_slice[2]],
                                                       color=list_of_color[i],
                                                       s=size_data["sc"],
                                                       alpha=list_of_alpha[i],
                                                       label=patches[i]
                                                       )

                        # recompute the ax.dataLim
                        axes[tuple(ind_slice)].relim()
                        # update ax.viewLim using the new dataLim
                        axes[tuple(ind_slice)].autoscale()

                idx_1 = ["x", "y", "z", "vx", "vy", "vz"].index(ind_slice[0])
                idx_2 = ["x", "y", "z", "vx", "vy", "vz"].index(ind_slice[1])
                idx_3 = ["x", "y", "z", "vx", "vy", "vz"].index(ind_slice[2])
                for i, sc in enumerate(self.lst_spec_sc):
                    axes[tuple(ind_slice)].scatter(sc.trajectory_track[idx_1][index],
                                                   sc.trajectory_track[idx_2][index],
                                                   sc.trajectory_track[idx_3][index],
                                                   color=colors[i] if sc.plot_color == "map" else sc.plot_color,
                                                   s=size_data["sc"],
                                                   label=sc.display_name
                                                   )

                    axes[tuple(ind_slice)].relim()
                    # update ax.viewLim using the new dataLim
                    axes[tuple(ind_slice)].autoscale()

                if ind_slice[0] in ["x", "y", "z"]:
                    xl = ind_slice[0] + " [m]"
                else:
                    xl = ind_slice[0] + " [m/s]"

                if ind_slice[1] in ["x", "y", "z"]:
                    yl = ind_slice[1] + " [m]"
                else:
                    yl = ind_slice[1] + " [m/s]"

                if ind_slice[2] in ["x", "y", "z"]:
                    zl = ind_slice[2] + " [m]"
                else:
                    zl = ind_slice[2] + " [m/s]"

                axes[tuple(ind_slice)].set_xlabel(xl)
                axes[tuple(ind_slice)].set_ylabel(yl)
                axes[tuple(ind_slice)].set_zlabel(zl)
                #fig.tight_layout()

                axes[tuple(ind_slice)].xaxis.set_pane_color(color_data["background"])
                axes[tuple(ind_slice)].yaxis.set_pane_color(color_data["background"])
                axes[tuple(ind_slice)].zaxis.set_pane_color(color_data["background"])

                axes[tuple(ind_slice)].set_facecolor(color_data["background"])

                lgnd = axes[tuple(ind_slice)].legend(loc='upper left')
                lgnd.set_draggable(True)
                for handle in lgnd.legend_handles:
                    handle.set_sizes([50])

    def trajectory_xyz(self):
        fig = plt.figure(self.figure_counter + 1)
        fig.set_facecolor(color_data["background"])
        self.figure_counter += 1

        n_samp = len(self.lst_spec_sc)
        cmap = plt.colormaps[color_data["map"]]
        colors = cmap(np.linspace(0, 1, n_samp))

        ax_1 = fig.add_subplot(231)
        ax_2 = fig.add_subplot(232)
        ax_3 = fig.add_subplot(233)
        ax_4 = fig.add_subplot(234)
        ax_5 = fig.add_subplot(235)
        ax_6 = fig.add_subplot(236)

        axes = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6]

        for ax in axes:
            ax.xaxis.label.set_color(color_data["ticks"])
            ax.yaxis.label.set_color(color_data["ticks"])

            ax.tick_params(axis='x', colors=color_data["ticks"])
            ax.tick_params(axis='y', colors=color_data["ticks"])

            ax.spines['left'].set_color(color_data["ticks"])
            ax.spines['top'].set_color(color_data["ticks"])
            ax.spines['right'].set_color(color_data["ticks"])
            ax.spines['bottom'].set_color(color_data["ticks"])

        patches = ["Boundary", "Center", "Generic", "Feasible"]

        list_of_dict, list_of_color, _ = plots_additional.full_trajectory_group(self.list_of_spacecraft)

        for i, ax in enumerate(axes):
            for g, group in enumerate(list_of_dict):
                if group["x"]:
                    ax.scatter(group["t"],
                               group[list(group.keys())[i]],
                               color=list_of_color[g],
                               s=1,
                               label=patches[g]
                               )
            ax.grid(visible=True, color=color_data["grid"])

        for i, ax in enumerate(axes):
            for sci, sc in enumerate(self.lst_spec_sc):
                if isinstance(sc.plot_color, str):
                    ax.scatter(self.integration_points,
                               sc.trajectory_track[i],
                               color=colors[sci],
                               s=1,
                               label=sc.display_name)
                    ax.plot(self.integration_points,
                            sc.trajectory_track[i],
                            color=colors[sci],
                            linewidth=size_data["dia_linewidth"],
                            alpha=size_data["plot_alpha"])
                else:
                    ax.scatter(self.integration_points,
                               sc.trajectory_track[i],
                               color=sc.plot_color,
                               s=1,
                               label=sc.display_name)
                    ax.plot(self.integration_points,
                            sc.trajectory_track[i],
                            color=sc.plot_color,
                            linewidth=size_data["dia_linewidth"],
                            alpha=size_data["plot_alpha"])

        ax_6.legend()
        lgnd = ax_6.legend()
        lgnd.set_draggable(True)
        for handle in lgnd.legend_handles:
            handle.set_sizes([50])

        for axis in axes:
            axis.set_xlabel("Time [s]")
            axis.set_facecolor(color_data["background"])

        ax_1.set_ylabel("X position [m]")
        ax_2.set_ylabel("Y position [m]")
        ax_3.set_ylabel("Z position [m]")
        ax_4.set_ylabel("X velocity [m/s]")
        ax_5.set_ylabel("Y velocity [m/s]")
        ax_6.set_ylabel("Z velocity [m/s]")

    def parameters_plot(self):
        fig = plt.figure(self.figure_counter + 1)
        self.figure_counter += 1
        fig.set_facecolor(color_data["background"])

        n_samp = len(self.lst_spec_sc)
        cmap = plt.colormaps[color_data["map"]]
        colors = cmap(np.linspace(0, 1, n_samp))

        ax_1 = fig.add_subplot(231)
        ax_2 = fig.add_subplot(232)
        ax_3 = fig.add_subplot(233)
        ax_4 = fig.add_subplot(234)
        ax_5 = fig.add_subplot(235)
        ax_6 = fig.add_subplot(236)

        axes = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6]

        for ax_1 in axes:
            ax_1.xaxis.label.set_color(color_data["ticks"])
            ax_1.yaxis.label.set_color(color_data["ticks"])

            ax_1.tick_params(axis='x', colors=color_data["ticks"])
            ax_1.tick_params(axis='y', colors=color_data["ticks"])

            ax_1.spines['left'].set_color(color_data["ticks"])
            ax_1.spines['top'].set_color(color_data["ticks"])
            ax_1.spines['right'].set_color(color_data["ticks"])
            ax_1.spines['bottom'].set_color(color_data["ticks"])

        patches = ["Boundary", "Center", "Generic", "Feasible"]

        list_of_dict, list_of_color, _ = plots_additional.full_kepler_parameter_group(self.list_of_spacecraft)

        for i, ax in enumerate(axes):
            for g, group in enumerate(list_of_dict):
                if group["SMA"]:
                    ax.scatter(group["t"],
                               group[list(group.keys())[i]],
                               color=list_of_color[g],
                               s=1,
                               label=patches[g]
                               )
            ax.grid(visible=True, color=color_data["grid"])
            ax.set_facecolor(color_data["background"])

        for i, ax in enumerate(axes):
            for sci, sc in enumerate(self.lst_spec_sc):
                if isinstance(sc.plot_color, str):
                    ax.scatter(self.integration_points,
                               sc.orbital_parameters_track[i],
                               color=colors[sci],
                               s=1,
                               label=sc.display_name)
                    ax.plot(self.integration_points,
                            sc.orbital_parameters_track[i],
                            color=colors[sci],
                            linewidth=size_data["dia_linewidth"],
                            alpha=size_data["plot_alpha"])
                else:
                    ax.scatter(self.integration_points,
                               sc.orbital_parameters_track[i],
                               color=sc.plot_color,
                               s=1,
                               label=sc.display_name)
                    ax.plot(self.integration_points,
                            sc.orbital_parameters_track[i],
                            color=sc.plot_color,
                            linewidth=size_data["dia_linewidth"],
                            alpha=size_data["plot_alpha"])

        ax_6.legend()
        lgnd = ax_6.legend()
        lgnd.set_draggable(True)
        for handle in lgnd.legend_handles:
            handle.set_sizes([50])

        for axis in axes:
            axis.set_xlabel("Time [s]")
            axis.set_facecolor(color_data["background"])

        ax_1.set_ylabel("SMA [m]")
        ax_2.set_ylabel("ECC [-]")
        ax_3.set_ylabel("INC [rad]")
        ax_4.set_ylabel("RAAN [rad]")
        ax_5.set_ylabel("APERI [rad]")
        ax_6.set_ylabel("TAEPO [rad]")

    def C3_plot(self):
        fig = plt.figure(self.figure_counter + 1, figsize=(7.5, 2.5 * 1.5))
        self.figure_counter += 1
        fig.set_facecolor(color_data["background"])

        list_of_dict, list_of_color, list_of_alpha = plots_additional.full_scalar_arrays_group(self.list_of_spacecraft,
                                                                                               ["C3"])

        ax_1 = fig.add_subplot(111)
        ax_1.set_title("Characteristic energy")

        ax_1.xaxis.label.set_color(color_data["ticks"])
        ax_1.yaxis.label.set_color(color_data["ticks"])

        ax_1.tick_params(axis='x', colors=color_data["ticks"])
        ax_1.tick_params(axis='y', colors=color_data["ticks"])

        ax_1.spines['left'].set_color(color_data["ticks"])
        ax_1.spines['top'].set_color(color_data["ticks"])
        ax_1.spines['right'].set_color(color_data["ticks"])
        ax_1.spines['bottom'].set_color(color_data["ticks"])

        ax_1.grid(visible=True, color=color_data["grid"])

        axes = [ax_1]
        patches = ["Boundary", "Center", "Generic", "Feasible"]

        n_samp = len(self.lst_spec_sc)
        cmap = plt.colormaps[color_data["map"]]
        colors = cmap(np.linspace(0, 1, n_samp))

        for i, ax in enumerate(axes):
            for g, group in enumerate(list_of_dict):
                if group["C3"]:
                    ax.scatter(group["t"],
                               group["C3"],
                               color=list_of_color[g],
                               s=1,
                               label=patches[g]
                               )

            for sci, sc_special in enumerate(self.lst_spec_sc):
                if isinstance(sc_special.plot_color, str):
                    ax.scatter(self.integration_points, sc_special.C3_track, s=1, color=colors[sci],
                               label=sc_special.display_name)
                    ax.plot(self.integration_points, sc_special.C3_track, color=colors[sci],
                            linewidth=size_data["dia_linewidth"], alpha=size_data["plot_alpha"])

                else:
                    ax.scatter(self.integration_points, sc_special.C3_track, s=1, color=sc_special.plot_color,
                               label=sc_special.display_name)
                    ax.plot(self.integration_points, sc_special.C3_track, color=sc_special.plot_color,
                            linewidth=size_data["dia_linewidth"], alpha=size_data["plot_alpha"])
            lgnd = plt.legend()
            for handle in lgnd.legend_handles:
                handle.set_sizes([50])
            lgnd.set_draggable(True)

        for axis in axes:
            axis.set_xlabel("Time [s]")
            axis.set_facecolor(color_data["background"])

        ax_1.set_ylabel("Characteristic energy [m^2 / s^2]")

    def magnitude_plot(self):
        fig = plt.figure(self.figure_counter + 1)
        fig.set_facecolor(color_data["background"])
        self.figure_counter += 1

        ax_1 = fig.add_subplot(121)
        ax_2 = fig.add_subplot(122)
        ax_1.set_title("Velocity magnitude")
        ax_2.set_title("Position magnitude")
        axes = [ax_1, ax_2]
        patches = ["Boundary", "Center", "Generic", "Feasible"]

        list_of_dict, list_of_color, list_of_alpha = plots_additional.full_scalar_arrays_group(self.list_of_spacecraft,
                                                                                               ["vel", "pos"])

        n_samp = len(self.lst_spec_sc)
        cmap = plt.colormaps[color_data["map"]]
        colors = cmap(np.linspace(0, 1, n_samp))

        # Full group
        for g, group in enumerate(list_of_dict):
            if group["vel"]:
                ax_1.scatter(group["t"],
                             group["vel"],
                             color=list_of_color[g],
                             s=1,
                             label=patches[g]
                             )

                ax_2.scatter(group["t"],
                             group["pos"],
                             color=list_of_color[g],
                             s=1,
                             label=patches[g]
                             )

        # Special spacecraft
        for sci, sc in enumerate(self.lst_spec_sc):
            if isinstance(sc.plot_color, str):
                ax_1.scatter(self.integration_points,
                             sc.vel_mag_track,
                             color=colors[sci],
                             s=1,
                             label=sc.display_name)

                ax_1.plot(self.integration_points,
                          sc.vel_mag_track,
                          color=colors[sci],
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])

                ax_2.scatter(self.integration_points,
                             sc.slant_range_track,
                             color=colors[sci],
                             s=1,
                             label=sc.display_name)
                ax_2.plot(self.integration_points,
                          sc.slant_range_track,
                          color=colors[sci],
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])
            else:
                ax_1.scatter(self.integration_points,
                             sc.vel_mag_track,
                             color=sc.plot_color,
                             s=1,
                             label=sc.display_name)
                ax_1.plot(self.integration_points,
                          sc.vel_mag_track,
                          color=sc.plot_color,
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])

                ax_2.scatter(self.integration_points,
                             sc.slant_range_track,
                             color=sc.plot_color,
                             s=1,
                             label=sc.display_name)
                ax_2.plot(self.integration_points,
                          sc.slant_range_track,
                          color=sc.plot_color,
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])

        for axis in axes:
            axis.set_xlabel("Time [s]")
            # axis.set_facecolor('xkcd:midnight')
            axis.grid(visible=True, color=[0.5, 0.5, 1])
            axis.set_facecolor(color_data["background"])

        ax_1.set_ylabel("Velocity magnitude [m]")
        ax_2.set_ylabel("Position magnitude [m/2]")

        ax_1.legend()
        lgnd = ax_1.legend()
        lgnd.set_draggable(True)
        for handle in lgnd.legend_handles:
            handle.set_sizes([50])

        ax_2.legend()
        lgnd = ax_2.legend()
        lgnd.set_draggable(True)
        for handle in lgnd.legend_handles:
            handle.set_sizes([50])

    def body_distances_plot(self, body_list):
        fig = plt.figure(self.figure_counter + 1, figsize=(6, 4.5))
        fig.set_facecolor(color_data["background"])
        self.figure_counter += 1

        n_samp = len(self.lst_spec_sc)
        cmap = plt.colormaps[color_data["map"]]
        colors = cmap(np.linspace(0, 1, n_samp))

        patches = ["Boundary", "Center", "Generic", "Feasible"]

        # Get the maximum required number of plots of the body trajectories
        body_plots = []
        for sc in self.list_of_spacecraft:
            # sc_body_plots = sc.body_distances.keys()
            for body in body_list:
                if body not in body_plots:
                    body_plots.append(body)

        for sc in self.lst_spec_sc:
            # sc_body_plots = sc.body_distances.keys()
            for body in body_list:
                if body not in body_plots:
                    body_plots.append(body)

        # Getting the arrangement of the subplots based on the number of bodies:
        if len(body_plots) == 1:
            height = 1
            width = 1
        if len(body_plots) == 2:
            height = 1
            width = 2
        if len(body_plots) > 2:
            height = 2
            width = math.ceil(len(body_plots) / 2)

        axes = dict()

        for k, body in enumerate(body_plots):
            axis = fig.add_subplot(height, width, k + 1)
            axis.set_title(body)
            axis.set_xlabel("Time [s]")
            axis.set_ylabel("Distance [m]")
            # axis.patch.set_facecolor('xkcd:midnight')
            axis.grid(visible=True, color=[0.5, 0.5, 1])
            axes[body] = axis

        if self.list_of_spacecraft:
            list_of_dict, list_of_color, list_of_alpha = plots_additional.full_body_dist_groups(self.list_of_spacecraft)

            for g, group in enumerate(list_of_dict):
                for body in body_list:
                    if group[body]:
                        axes[body].scatter(group["t"],
                                           group[body],
                                           color=list_of_color[g],
                                           s=1,
                                           label=patches[g]
                                           )
                        lgnd = axes[body].legend()
                        lgnd.set_draggable(True)
                        for handle in lgnd.legend_handles:
                            handle.set_sizes([50])

        if self.lst_spec_sc:
            for body in body_list:
                for sci, sc_special in enumerate(self.lst_spec_sc):
                    if isinstance(sc_special.plot_color, str):
                        axes[body].scatter(self.integration_points,
                                           sc_special.body_distances[body],
                                           color=colors[sci],
                                           s=1,
                                           label=sc_special.display_name)
                        axes[body].plot(self.integration_points,
                                        sc_special.body_distances[body],
                                        color=colors[sci],
                                        linewidth=size_data["dia_linewidth"],
                                        alpha=size_data["plot_alpha"])
                    else:
                        axes[body].scatter(self.integration_points,
                                           sc_special.body_distances[body],
                                           color=sc_special.plot_color,
                                           s=1,
                                           label=sc_special.display_name)
                        axes[body].plot(self.integration_points,
                                        sc_special.body_distances[body],
                                        color=sc_special.plot_color,
                                        linewidth=size_data["dia_linewidth"],
                                        alpha=size_data["plot_alpha"])
                lgnd = axes[body].legend()
                lgnd.set_draggable(True)
                for handle in lgnd.legend_handles:
                    handle.set_sizes([50])

                axes[body].grid(visible=True, color=[0.5, 0.5, 1])
                axes[body].set_facecolor(color_data["background"])

    def sensitivity_plot(self, diag_only=False):
        fig = plt.figure(self.figure_counter + 1, figsize=(7.5, 2.5 * 1.5))
        fig.set_facecolor(color_data["background"])
        self.figure_counter += 1
        ax_1 = fig.add_subplot(111)
        ax_1.set_title("Sensitivity analysis")

        textmark = ["dx", "dy", "dz", "dvx", "dvy", "dvz"]

        if diag_only:
            cmap = plt.colormaps[color_data["map"]]
            colors = cmap(np.linspace(0, 1, 6))

            for i in range(6):
                ax_1.plot(self.integration_points, self.sensitivity[i][i], color=colors[i],
                          label=textmark[i] + "/" + textmark[i])

            plt.legend()
            ax_1.grid(visible=True, color=[0.5, 0.5, 1])
            ax_1.set_ylabel("Sensitivity")
            ax_1.set_xlabel("Time [s]")
        else:
            cmap_1 = plt.colormaps[color_data["map"]]
            colors_1 = cmap_1(np.linspace(0, 1, 6))

            cmap_2 = plt.colormaps["cool"]
            colors_2 = cmap_2(np.linspace(0, 1, 30))

            idx_1 = 0
            idx_2 = 0
            firstcall = True
            for i in range(6):
                for j in range(6):
                    if i == j:
                        ax_1.plot(self.integration_points, self.sensitivity[i][j], color=colors_1[idx_1],
                                  label=textmark[i] + "/" + textmark[j])
                        idx_1 += 1
                    else:
                        if firstcall:
                            ax_1.plot(self.integration_points, self.sensitivity[i][j], color=colors_2[idx_1],
                                      label="Cross-coupling", linestyle=(0, (5, 5)))
                            firstcall = False
                        else:
                            ax_1.plot(self.integration_points, self.sensitivity[i][j], color=colors_2[idx_1],
                                      linestyle=(0, (5, 5)))
                        idx_2 += 1

            plt.legend()
            ax_1.grid(visible=True, color=[0.5, 0.5, 1])
            ax_1.set_ylabel("Sensitivity")
            ax_1.set_xlabel("Time [s]")
            ax_1.set_facecolor(color_data["background"])

    def plot_steering(self):
        fig = plt.figure(self.figure_counter + 1, figsize=(7.5, 2.5 * 1.5))
        fig.set_facecolor(color_data["background"])
        self.figure_counter += 1

        ax_1 = fig.add_subplot(411)
        ax_1.set_title("Control inertial x")

        ax_2 = fig.add_subplot(412)
        ax_2.set_title("Control inertial y")

        ax_3 = fig.add_subplot(413)
        ax_3.set_title("Control inertial z")

        ax_4 = fig.add_subplot(414)
        ax_4.set_title("Control inertial magnitude")

        n_samp = len(self.lst_spec_sc)
        cmap = plt.colormaps[color_data["map"]]
        colors = cmap(np.linspace(0, 1, n_samp))

        for sci, sc_special in enumerate(self.lst_spec_sc):

            if isinstance(sc_special.plot_color, str):
                ax_1.scatter(self.integration_points,
                                   sc_special.steer_x,
                                   color=colors[sci],
                                   s=1,
                                   label=sc_special.display_name)
                ax_1.plot(self.integration_points,
                                sc_special.steer_x,
                                color=colors[sci],
                                linewidth=size_data["dia_linewidth"],
                                alpha=size_data["plot_alpha"])

                ax_2.scatter(self.integration_points,
                             sc_special.steer_y,
                             color=colors[sci],
                             s=1,
                             label=sc_special.display_name)
                ax_2.plot(self.integration_points,
                          sc_special.steer_y,
                          color=colors[sci],
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])

                ax_3.scatter(self.integration_points,
                             sc_special.steer_z,
                             color=colors[sci],
                             s=1,
                             label=sc_special.display_name)
                ax_3.plot(self.integration_points,
                          sc_special.steer_z,
                          color=colors[sci],
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])

                ax_4.scatter(self.integration_points,
                             sc_special.steer_magnitude,
                             color=colors[sci],
                             s=1,
                             label=sc_special.display_name)
                ax_4.plot(self.integration_points,
                          sc_special.steer_magnitude,
                          color=colors[sci],
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])
            else:
                ax_1.scatter(self.integration_points,
                             sc_special.steer_x,
                             color=sc_special.plot_color,
                             s=1,
                             label=sc_special.display_name)
                ax_1.plot(self.integration_points,
                          sc_special.steer_x,
                          color=sc_special.plot_color,
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])

                ax_2.scatter(self.integration_points,
                             sc_special.steer_y,
                             color=sc_special.plot_color,
                             s=1,
                             label=sc_special.display_name)
                ax_2.plot(self.integration_points,
                          sc_special.steer_y,
                          color=sc_special.plot_color,
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])

                ax_3.scatter(self.integration_points,
                             sc_special.steer_z,
                             color=sc_special.plot_color,
                             s=1,
                             label=sc_special.display_name)
                ax_3.plot(self.integration_points,
                          sc_special.steer_z,
                          color=sc_special.plot_color,
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])

                ax_4.scatter(self.integration_points,
                             sc_special.steer_magnitude,
                             color=sc_special.plot_color,
                             s=1,
                             label=sc_special.display_name)
                ax_4.plot(self.integration_points,
                          sc_special.steer_magnitude,
                          color=sc_special.plot_color,
                          linewidth=size_data["dia_linewidth"],
                          alpha=size_data["plot_alpha"])

        axes = [ax_1, ax_2, ax_3, ax_4]
        for axis in axes:
            axis.set_xlabel("Time [s]")
            axis.set_ylabel("Aceleration [m/s^2]")
            axis.grid(visible=True, color=[0.5, 0.5, 1])


            axis.grid(visible=True, color=[0.5, 0.5, 1])
            axis.set_facecolor(color_data["background"])

        lgnd = ax_3.legend()
        lgnd.set_draggable(True)
        for handle in lgnd.legend_handles:
            handle.set_sizes([50])

        fig.subplots_adjust(hspace=0.5)

