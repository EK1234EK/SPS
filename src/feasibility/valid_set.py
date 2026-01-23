import math
import matplotlib.pyplot as plt
import src.feasibility.inequality_conditions as ic
import src.feasibility.planar_condition as pc
from src.globals import colors
import numpy as np
from src.spacecraft import sc
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Point, Polygon
import random

color_data, _ = colors.get_color_data()

class feasibility_setup:

    def __init__(self, list_of_sc, force_model, manifolds):
        self.list_of_sc = list_of_sc
        self.feasibility_condition = -1
        self.central_mass = force_model.central_mass
        self.force_model = force_model
        self.manifolds = manifolds

        self.sample_points = []
        self.fill_num = 0
        self.return_type = None  # Use "base", "new", "all" or None
        self.resampled_s_vecs = dict()
        self.resampled_dist = dict()
        self.resample_edge_only = True
        self.cutoff_dist = None

        self.convex_fill_points = 100

        self.normal = np.array([])
        self.origin = np.array([])
        self.resolve_trajectory()

        self.list_of_res_sc = []

    def check_conditions(self):
        for sc in self.list_of_sc:
            if ic.check_and(sc):
                # print("Condition 1 triggered")
                sc.plot_color = color_data["feasible"]
                sc.is_feasible = True
            else:
                # sc.plot_color = "map"
                sc.is_feasible = False

    def check_planar_conditions(self, naive_sampling=True, full_projection=True, discretization=10):
        self.resampled_s_vecs = dict()
        self.normal = self.normal / (np.linalg.norm(self.normal))
        print(self.normal)
        for sc in self.list_of_sc:
            sc_is_valid, sc_resample_dict, dist_dict = pc.check_conditions(sc=sc,
                                                                origin=self.origin,
                                                                normal=self.normal,
                                                                sample_points=self.sample_points,
                                                                fill_num=self.fill_num,
                                                                return_type=self.return_type,
                                                                cutoff_dist=self.cutoff_dist)
            if sc_is_valid:
                sc.plot_color = [0, 1, 1]
                sc.is_feasible = True
                if self.return_type is not None:
                    # In case resample_edge_only == True, only edge spacecraft get interpolated
                    if self.resample_edge_only and sc.is_edge_spacecraft:
                        self.resampled_s_vecs = self.merge_resample_dicts(sc_resample_dict, self.resampled_s_vecs)
                        self.resampled_dist = self.merge_resample_dicts(dist_dict, self.resampled_dist)
                    elif not self.resample_edge_only:
                        self.resampled_s_vecs = self.merge_resample_dicts(sc_resample_dict, self.resampled_s_vecs)
                        self.resampled_dist = self.merge_resample_dicts(dist_dict, self.resampled_dist)
            else:
                sc.plot_color = "map"
                sc.is_feasible = False

        if int(input("Project onto plane?")) == 1:
            self.project_to_plane(full_projection=full_projection, naive_sampling=naive_sampling, discretization=discretization)
        pass

    def merge_resample_dicts(self, dict_1: dict, dict_2: dict):
        for key in dict_1.keys():
            if key in dict_2.keys():
                dict_2[key] += dict_1[key]
            else:
                dict_2[key] = dict_1[key]

        return dict_2


    def resolve_trajectory(self):
        for sc in self.list_of_sc:
            if len(sc.integration_points) == 0:
                sc.trajectory_conversion(mass=self.central_mass)


    def create_resampled_sc(self):
        if len(self.resampled_s_vecs.keys()) == 0:
            print("Warning! No resampled states available! Nothing done!")
        else:
            # Loop over all timestamps
            for ts in self.resampled_s_vecs.keys():
                # Loop over all state vectors in each timegroup
                for svec in self.resampled_s_vecs[ts]:
                    sc_new = sc.Spacecraft(init_state_vector=svec, force_model=self.force_model)
                    sc_new.time_interval = [ts, 0]
                    sc_new.integration_points = np.linspace(ts, 0, 300).tolist()
                    sc_new.is_resampled_spacecraft = True
                    sc_new.integrate_states_sivp(rtol=1e-9)
                    sc_new.revert_trajectory()
                    # sc_new.integration_points = list(reversed(sc_new.integration_points))
                    self.list_of_res_sc.append(sc_new)

        self.remove_outbound_sc()

        # Remove all spacecraft that are outside the manifold


    ## The projection begins ##

    def project_to_plane(self, full_projection=False, naive_sampling=True, discretization=10):

        # Selecting reference frame vectors:
        keys = list(self.resampled_s_vecs.keys())
        if len(keys) != 1:
            print("Warning! More than one timestamp in resampled state vectors!", keys)
            idx = int(input("Specify timestamp index!"))
            key = keys[idx]
        else: key = keys[0]

        points = self.resampled_s_vecs[key]
        """for i in range(5):
            print(points[i+50])
            print()"""
        dist = self.resampled_dist[key]

        if not full_projection:
            # TESTING: Also projecting the off-plane points onto the plane, but keeping the entry in the dist array
            for p, point in enumerate(points):
                if dist[p] > 1:
                    points[p] = (np.array(point) - (1 / (np.linalg.norm(self.normal))) * self.normal * dist[p]).tolist()

            # Selecting the second axis
            axis_1 = points[0]
            axis_2 = None
            for point in points:
                for k in range(len(point)):
                    if (axis_1[k] != 0) and (point[k] != 0) and ((axis_1[k] / point[k] < 0.99) or (axis_1[k] / point[k] > 1.01)) and ((axis_1[k] / point[k] > -0.99) or (axis_1[k] / point[k] < -1.01)):
                        axis_2 = point
                        break
            if axis_2 is None: return None

            span_ang = math.acos((np.dot(np.array(axis_1), np.array(axis_2))) / (np.linalg.norm(np.array(axis_1)) * np.linalg.norm(axis_2)))

            # Selecting suitable dimensions for k-solving
            axes = [0, 0]
            for i in range(len(axis_1)):
                if axis_1[i] != 0 and axis_2[i] != 0:
                    axes[0] = i
            for i in range(len(axis_1)):
                if i != axes[0] and axis_1[i] != axis_2[i]:
                    axes[1] = i

            # The matrix used for
            mat_inv = np.linalg.inv(np.array([[axis_1[axes[0]], axis_2[axes[0]]], [axis_1[axes[1]], axis_2[axes[1]]]]))

            rot_mat = np.array([[math.cos(span_ang), math.sin(span_ang)], [-math.sin(span_ang), math.cos(span_ang)]])

            # Getting the k-factors of all the points
            # sol = [dict() for _ in range(len(points))]
            x_t = [0.0 for _ in range(len(points))]
            y_t = [0.0 for _ in range(len(points))]

            t_vec_1 = [np.linalg.norm(np.array(axis_1)), 0]
            t_vec_2 = np.dot(rot_mat, np.array([np.linalg.norm(np.array(axis_2)), 0]))

            for p, point in enumerate(points):
                k = np.dot(mat_inv, np.array([[point[axes[0]]], [point[axes[1]]]]))
                # Getting the tranformed solution

                p_t = k[0] * t_vec_1 + k[1] * t_vec_2
                x_t[p] = p_t[0]
                y_t[p] = p_t[1]

            if len(x_t) == 2:
                """x_t += [(x_t[0] + x_t[1]) * 0.501, (x_t[0] + x_t[1]) * 0.499]
                y_t += [(y_t[0] + y_t[1]) * 0.499, (y_t[0] + y_t[1]) * 0.501]
                dist += [(dist[0] + dist[1]) * 0.5, (dist[0] + dist[1]) * 0.5]"""
                x_res = np.linspace(x_t[0], x_t[1], self.convex_fill_points)
                y_res = np.linspace(y_t[0], y_t[1], self.convex_fill_points)

            else:

                x_t0 = []
                y_t0 = []
                z_t0 = []

                x_t1 = []
                y_t1 = []
                z_t1 = []
                for i in range(len(x_t)):
                    if dist[i] == 0:
                        x_t0.append(x_t[i])
                        y_t0.append(y_t[i])
                        z_t0.append(dist[i])

                    else:
                        x_t1.append(x_t[i])
                        y_t1.append(y_t[i])
                        z_t1.append(dist[i])

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                # Now set color to white (or whatever is "invisible")
                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')
                ax.scatter(x_t0, y_t0, z_t0, alpha=1, color=[0.8, 0, 0.8], s=15, label="Projection")
                ax.scatter(x_t1, y_t1, z_t1, alpha=1, color=[0, 0.4, 0], s=15, label="True distance")
                ax.set_xlabel("Dim 1")
                ax.set_ylabel("Dim 2")
                ax.set_zlabel("Distance")
                ax.grid(visible=True, color=[0.5, 0.5, 1])
                ax.legend()
                ax.set_title("Projection onto 6d hyperplane")
                plt.show()

                cont = bool(input("Continue with convex hull meshing?"))
                if not cont:
                    exit("Aborted before meshing")
                if naive_sampling:
                    x_res, y_res = self.mesh_convex_set(x_t, y_t)
                else:
                    points = [(x_t[i], y_t[i]) for i in range(len(x_t))]
                    hull = ConvexHull(points)

                    ordered_corners = [points[hull.vertices[i]] for i in range(len(hull.vertices))]

                    xl = [ordered_corners[i][0] for i in range(len(ordered_corners))]
                    yl = [ordered_corners[i][1] for i in range(len(ordered_corners))]

                    res_Vec = self.points_in_hull([[xl[idx], yl[idx]] for idx in range(len(xl))], 10)

                    x_res = [Vec[0] for Vec in res_Vec]
                    y_res = [Vec[1] for Vec in res_Vec]

                    fig = plt.figure(figsize=(7.5, 2.5 * 1.5))
                    ax = fig.add_subplot()
                    ax.scatter(x_t0, y_t0, alpha=1, color=[0.7, 0, 0.7], s=15, label="Projection")
                    ax.scatter(x_res, y_res, color=[1, 0, 0], s=10, label="Resampled points")
                    ax.scatter(xl, yl, color=[0, 0, 1], s=30, label="Convex vertices")
                    ax.plot(xl + [xl[0]], yl + [yl[0]], color=[1, 0, 0], label="Convex hull")
                    ax.set_xlabel("Dim 1")
                    ax.set_ylabel("Dim 2")
                    ax.grid(visible=True, color=[0.5, 0.5, 1])
                    if naive_sampling:
                        ax.set_title("Resampled interior, naive sampling")
                    else:
                        ax.set_title("Resampled interior, vertex-vertex linear")
                    ax.legend()
                    plt.show()

                print("Transforming back to 6d state space")
                # Getting the k-factors for each point

            backrot_mat_inv = np.linalg.inv(np.array([[t_vec_1[0], t_vec_2[0]], [t_vec_1[1], t_vec_2[1]]]))

            trans_points = [[] for _ in range(len(x_res))]
            for i in range(len(x_res)):
                dot_vec = np.array([x_res[i], y_res[i]])
                k_vec = np.dot(backrot_mat_inv, dot_vec)
                # Getting point in 6d space
                trans_points[i] = (k_vec[0] * np.array(axis_1) + k_vec[1] * np.array(axis_2)).tolist()

            # Ladies and gentlemen, we got 'em
            self.resampled_s_vecs[key] = trans_points

        else:
            # Rotation matrix to subspace


            # There is no turning back
            # hull = ConvexHull(points, qhull_options="QJ")
            self.resampled_s_vecs[key] = self.Random_Points_in_Nd_hull(points, naive_sampling, discretization)


    def mesh_convex_set(self, x, y):
        # Get the points as a list of tuples:
        points = [(x[i], y[i]) for i in range(len(x))]
        hull = ConvexHull(points)

        ordered_corners = [points[hull.vertices[i]] for i in range(len(hull.vertices))]

        xl = [ordered_corners[i][0] for i in range(len(ordered_corners))]
        yl = [ordered_corners[i][1] for i in range(len(ordered_corners))]

        fig = plt.figure(figsize=(7.5, 2.5*1.5))
        ax = fig.add_subplot()
        ax.scatter(x, y, color=[0, 0.8, 0], s=15, label="Projected points")
        ax.scatter(xl, yl, color=[0, 0, 1], s=30, label="Convex vertices")
        ax.plot(xl + [xl[0]], yl + [yl[0]], color=[1, 0, 0], label="Convex hull")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(visible=True, color=[0.5, 0.5, 1])
        ax.set_title("Convex hull")
        ax.legend()
        plt.show()


        # Resample convex polygon
        polygon = Polygon(ordered_corners)
        points = self.Random_Points_in_Polygon(polygon, self.convex_fill_points)
        xp, yp = polygon.exterior.xy

        x_res = [point.x for point in points]
        y_res = [point.y for point in points]

        fig = plt.figure(figsize=(7.5, 2.5*1.5))
        ax = fig.add_subplot()
        ax.scatter(x, y, color=[0, 0.7, 0], s=15, label="Projected points")
        ax.scatter(x_res, y_res, color=[1, 0, 0], s=10, label="Resampled points")
        ax.scatter(xl, yl, color=[0, 0, 1], s=30, label="Convex vertices")
        ax.plot(xl + [xl[0]], yl + [yl[0]], color=[1, 0, 0], label="Convex hull")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(visible=True, color=[0.5, 0.5, 1])
        ax.set_title("Resampled interior, naive sampling")
        ax.legend()
        plt.show()
        pass

        return x_res, y_res

    def Random_Points_in_Polygon(self, polygon, number):

        points = []

        minx, miny, maxx, maxy = polygon.bounds

        while len(points) < number:

            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))

            if polygon.contains(pnt):
                points.append(pnt)

        return points

    def Random_Points_in_Nd_hull(self, points, naive_sampling, discretization):

        rm_lines = dict()
        first_vec = points[0]
        reduced_normal = []

        for i in range(6):
            # Have some tolerance for numeric inaccuracy
            check = [True if ((points[k][i] >= (first_vec[i] - 0.0000001)) and (points[k][i] <= (first_vec[i] + 0.0000001))) else False for k in range(len(points))]
            if not False in check and self.normal[i] == 0:
                rm_lines[i] = first_vec[i]
            else:
                reduced_normal.append(self.normal[i])

        # Remove the lines which are planes of the subspace
        mod_points = [[points[i][k] for k in range(6) if k not in rm_lines.keys()] for i in range(len(points))]

        # Get the reduced target vector
        if not 0 in reduced_normal:
            raise ValueError("There is a non-zero component in the reduced normal vector! Modify hyperplane definition!")
        reduced_tgt = [0 for _ in range(len(reduced_normal))]
        for i in range(len(reduced_normal)):
            if reduced_normal[i] == 0:
                reduced_tgt[i] = 1
                break

        # Rotate the subspace to some reference orientation, such that the planar component can be identified
        R = self.rotmat_from_vec(target=reduced_tgt / np.linalg.norm(reduced_tgt), normal=reduced_normal / np.linalg.norm(reduced_normal))

        testvec = R.dot(reduced_normal)
        rot_points = [R.dot(vec) for vec in mod_points]
        # Removing the planar points
        rm_line_rotated = dict()

        first_vec_rot = rot_points[0]

        for i in range(len(first_vec_rot)):
            # Have some tolerance for numeric inaccuracy
            check = [True if ((rot_points[k][i] >= (first_vec_rot[i] - 0.0001)) and (rot_points[k][i] <= (first_vec_rot[i] + 0.0001))) else False for k in range(len(rot_points))]
            if not False in check:
                rm_line_rotated[i] = first_vec_rot[i]

        mod_rot_points = [[rot_points[i][k] for k in range(len(reduced_normal)) if k not in rm_line_rotated.keys()] for i in range(len(rot_points))]

        """if len(mod_rot_points[0]) == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(mod_points[0][:], mod_points[1][:], mod_points[2][:])
            plt.show()
        elif len(mod_rot_points[0]) == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(mod_points[0][:], mod_points[1][:])
            plt.show()"""

        # Now we create a Hull in the rotated subspace
        """ hull = Delaunay(mod_rot_points, qhull_options="Q12")

        maxlim = np.amax(hull.points, 0)
        minlim = np.amin(hull.points, 0)

        # Get volume fraction of hull
        full_hull = ConvexHull(hull.points)"""

        maxlim = np.amax(mod_rot_points, 0)
        minlim = np.amin(mod_rot_points, 0)
        full_hull = ConvexHull(mod_rot_points, qhull_options="Q12")

        verts = [full_hull.points[i] for i in full_hull.vertices]
        hull = Delaunay(verts)

        vol_subspace = 1
        for i in range(len(maxlim)):
            vol_subspace *= (maxlim[i] - minlim[i])

        print("Volume fraction of subspace: ", round((full_hull.volume / vol_subspace) * 100, 10), " %")

        resample_points = []

        if naive_sampling:

            while len(resample_points) < self.convex_fill_points:
                pnt = [np.random.uniform(minlim[i], maxlim[i]) for i in range(len(minlim))]

                if hull.find_simplex(pnt)>=0:
                    print("In hull")
                    resample_points.append(pnt)

        else:
            resample_points = self.points_in_hull(verts=hull.points, discretization=discretization)

        # Ok, now we insert the rotation-reduced dimension back into the thing
        # Merge the hull and the resampled points
        mod_rot_cloud = resample_points + mod_rot_points

        if len(mod_rot_cloud[0]) == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(mod_rot_cloud[0][:], mod_rot_cloud[1][:], mod_rot_cloud[2][:])
            plt.show()
        elif len(mod_rot_points[0]) == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(mod_rot_cloud[0][:], mod_rot_cloud[1][:])
            plt.show()

        rot_cloud = [[] for _ in range(len(mod_rot_cloud))]
        for k, point in enumerate(rot_cloud):
            for i in range(len(mod_rot_cloud[0])):
                if i in rm_line_rotated.keys():
                    point.append(rm_line_rotated[i])
                point.append(mod_rot_cloud[k][i])

        # Rotate the points back:
        R_inv = np.linalg.inv(R)
        cloud_mod = [[] for _ in range(len(mod_rot_cloud))]
        for i in range(len(mod_rot_cloud)):
            cloud_mod[i] = R_inv.dot(rot_cloud[i])

        # Reinsert the removed dimensions from earlier on
        cloud = [[_ for k in range(6)] for _ in range(len(cloud_mod))]
        for k, point in enumerate(cloud):
            ctr = 0
            for i in range(6):
                if i in rm_lines.keys():
                    point[i] = rm_lines[i]
                    ctr += 1
                else:
                    point[i] = cloud_mod[k][i - ctr]
        return cloud

    def remove_outbound_sc(self):
        rem = int(input("Remove spacecraft outside of manifolds?"))
        if rem != 1:
            return

        print("Removing spacecraft outside manifolds...")
        valid_sc = []
        for sc in self.list_of_res_sc:
            traj = sc.trajectory_track
            is_valid = True
            for i in range(6):
                if not (min([self.manifolds[i][0], self.manifolds[i][1]]) <= traj[i][0] <= max([self.manifolds[i][0], self.manifolds[i][1]])) and self.manifolds[i][2] != 1:
                    is_valid = False
            if is_valid:
                valid_sc.append(sc)
        self.list_of_res_sc = valid_sc
        pass

    def rotmat_from_vec(self, target, normal):

        # A subspace is defined by a normal vector n. n Indicates the orientation of the subspace in the global space.
        # The normal vector is therefore defined in the global space.
        # This function rotates points, which exist only in the subspace but have coordinates in the global space,
        # in such a way, that the linear constraint of the subspace is projected onto a fundamental plane. The orientation
        # of the fundamental plane is defined by the target vector. This algorithm determines a rotation matrix, required
        # to rotate a vector of the subspace to the fundamental plane. For this, "target" and "n" must form an
        # orthonormal basis. This can easily be achieved by having a zero entry in the normal vector and a 1 in the target
        # vector, with each other entry being zero.

        target = target / np.linalg.norm(target)
        normal = normal / np.linalg.norm(normal)
        ang = math.acos(np.dot(target, normal))
        n = len(list(target))
        unity = np.diag(v=[1 for _ in range(n)])
        t_1 = math.sin(ang) * (np.outer(target.T, normal) - np.outer(normal.T, target))
        t_2 = (math.cos(ang) - 1) * (np.outer(target.T, target) + np.outer(normal.T, normal))
        rotmat = unity + t_1 + t_2

        return rotmat

    def points_in_hull(self, verts, discretization=10):
        # Draw random lines between verticies
        resampled = []
        for k in range(math.floor(self.convex_fill_points / discretization)):
            a = random.choice(verts)
            b = random.choice(verts)
            while np.all(b == a):
                b = random.choice(verts)

            sample_vec = []
            for a_1, b_1 in zip(a, b):
                sample_vec.append(list(np.linspace(a_1, b_1, discretization)))

            for i in range(len(sample_vec[0])):
                new_vec = [sample_vec[p][i] for p in range(len(a))]
                resampled.append(new_vec)

        return resampled

    def transpose(self, array):
        array = array[:]  # make copy to avoid changing original
        n = len(array)
        for i, row in enumerate(array):
            array[i] = row + [None for _ in range(n - len(row))]

        array = zip(*array)

        for i, row in enumerate(array):
            array[i] = [elem for elem in row if elem is not None]

        return array
