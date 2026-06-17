import numpy as np
from matplotlib.pyplot import pause
import copy
from src.astrodynamic_functions.kepler_dynamics import time_to_true_anomaly
from src.spacecraft import sc
from src.guidance import steering_laws
import time
import multiprocessing


class particle_swarm:
    def __init__(self, manifolds, force_model):
        self.list_of_state_vectors = []
        self.list_of_edge_state_vectors = []
        self.list_of_center_state_vectors = []
        self.list_of_spacecraft = []
        self.list_of_edge_spacecraft = []
        self.list_of_center_spacecraft = []
        self.manifolds = manifolds  # -> Last manifold entry is the time
        self.force_model_1 = force_model  # Already has it's central attractor specified

        self.list_of_time_ivs = []
        self.list_of_edge_time_ivs = []
        self.list_of_center_time_ivs = []

        self.integration_points = []
        self.direct_transformation = ("Kepler", "C3", "State_magnitude")

        self.body_trajectories__safe = dict()

        self.sensitivity_delta = (100, 100, 100, 0.1, 0.1, 0.1)
        self.sens_mat = None

        # Only for spacecraft generation, not for actual integration
        self.do_integration = True

    def square_swarm(self, point_type: str = 'generic'):
        """
        :param point_type: str ('generic', 'edge', 'center')
        """

        discrete_arrays = []
        for k in range(7):
            discrete_arrays.append(
                [np.linspace(self.manifolds[k][0], self.manifolds[k][1], self.manifolds[k][2]).tolist()][0])

        for k_1 in discrete_arrays[0]:
            for k_2 in discrete_arrays[1]:
                for k_3 in discrete_arrays[2]:
                    for k_4 in discrete_arrays[3]:
                        for k_5 in discrete_arrays[4]:
                            for k_6 in discrete_arrays[5]:
                                for k_7 in discrete_arrays[6]:

                                    edge_state = False
                                    k = [k_1, k_2, k_3, k_4, k_5, k_6, k_7]
                                    for i in range(7):
                                        if ((k[i] == discrete_arrays[i][0]) or (
                                                k[i] == discrete_arrays[i][-1])) and len(
                                                discrete_arrays[i]) > 1:
                                            edge_state = True

                                    if edge_state and point_type == 'edge':
                                        self.list_of_edge_state_vectors.append([k_1, k_2, k_3, k_4, k_5, k_6])
                                        self.list_of_edge_time_ivs.append([k_7, self.integration_points[-1]])

                                    elif not edge_state and point_type == 'center':
                                        self.list_of_center_state_vectors.append([k_1, k_2, k_3, k_4, k_5, k_6])
                                        self.list_of_center_time_ivs.append([k_7, self.integration_points[-1]])

                                    elif point_type == 'generic':
                                        self.list_of_state_vectors.append([k_1, k_2, k_3, k_4, k_5, k_6])
                                        self.list_of_time_ivs.append([k_7, self.integration_points[-1]])
        pass

    def sensitivity_swarm(self, center):
        dx = self.sensitivity_delta
        self.list_of_state_vectors = [copy.deepcopy(center) for _ in range(7)]
        self.list_of_time_ivs = [[self.integration_points[0], self.integration_points[-1]]]
        for i in range(len(dx)):
            self.list_of_state_vectors[i + 1][i] += dx[i]
            self.list_of_time_ivs.append([self.integration_points[0], self.integration_points[-1]])

    def get_sensitivity(self):
        dx = self.sensitivity_delta
        sens = dict()
        sens["center"] = self.list_of_spacecraft[0].trajectory_track
        sens["+x"] = self.list_of_spacecraft[1].trajectory_track
        sens["+y"] = self.list_of_spacecraft[2].trajectory_track
        sens["+z"] = self.list_of_spacecraft[3].trajectory_track
        sens["+vx"] = self.list_of_spacecraft[4].trajectory_track
        sens["+vy"] = self.list_of_spacecraft[5].trajectory_track
        sens["+vz"] = self.list_of_spacecraft[6].trajectory_track

        # Now get the Jacobian

        mat = [[] for _ in range(len(dx))]
        for linenum, key in enumerate(["+x", "+y", "+z", "+vx", "+vy", "+vz"]):
            line = [[] for _ in range(len(dx))]
            for k in range(len(dx)):
                arr = [_ for _ in range(len(sens["+x"][0]))]
                for i in range(len(sens[key][0])):
                    arr[i] = (sens[key][k][i] - sens["center"][k][i]) / dx[linenum]

                line[k] = arr
            mat[linenum] = line
        self.sens_mat = mat

    def create_and_integrate_swarm(self, method='DOP853', rtol=1e-3, atol=1e-6, parproc=False, cores=4):

        self.method = method
        self.rtol = rtol
        self.atol = atol

        if parproc:
            print("Integrating states via parallel processing")
            Pool = multiprocessing.Pool(cores)

            # Beginning computation
            sc_generic = []
            if len(self.list_of_state_vectors) != 0:
                t_1 = time.time()
                print()
                print("=================================================")
                print()
                print("Integrating", len(self.list_of_state_vectors), " generic set of states")
                print()
                print("=================================================")
                print()
                sc_generic = Pool.map(self._integrate_generic_states, range(len(self.list_of_state_vectors)))

                while None in sc_generic:
                    sc_generic.remove(None)

                t_2 = time.time()
                print("Integrating generic states: " + str(t_2 - t_1) + " s")

            sc_edge = []
            if len(self.list_of_edge_state_vectors) != 0:
                t_1 = time.time()
                print()
                print("=================================================")
                print()
                print("Integrating", len(self.list_of_edge_state_vectors), " subset edge state vectors")
                print()
                print("=================================================")
                print()
                sc_edge = Pool.map(self._integrate_edge_states, range(len(self.list_of_edge_state_vectors)))

                while None in sc_edge:
                    sc_edge.remove(None)

                t_2 = time.time()
                print("Integrating edge states: " + str(t_2 - t_1) + " s")

            sc_center = []
            if len(self.list_of_center_state_vectors) != 0:
                t_1 = time.time()
                print()
                print("=================================================")
                print()
                print("Integrating", len(self.list_of_center_state_vectors), " subset center state vectors")
                print()
                print("=================================================")
                print()
                sc_center = Pool.map(self._integrate_center_states, range(len(self.list_of_center_state_vectors)))

                while None in sc_center:
                    sc_center.remove(None)

                t_2 = time.time()
                print("Integrating center states: " + str(t_2 - t_1) + " s")

            self.list_of_spacecraft = sc_generic + sc_center + sc_edge
            Pool.close()

        else:
            temp_lst = []

            print()
            print("=================================================")
            print()
            print("Integrating", len(self.list_of_state_vectors), " generic set of states")
            print()
            print("=================================================")
            print()
            t_1 = time.time()
            for i, state_vector in enumerate(self.list_of_state_vectors):
                temp_lst.append(self._integrate_generic_states(i))
            t_2 = time.time()
            print("Integrating generic states: " + str(t_2 - t_1) + " s")

            t_1 = time.time()
            print()
            print("=================================================")
            print()
            print("Integrating", len(self.list_of_edge_state_vectors), " subset edge state vectors")
            print()
            print("=================================================")
            print()
            for i, state_vector in enumerate(self.list_of_edge_state_vectors):
                temp_lst.append(self._integrate_edge_states(i))
            t_2 = time.time()
            print("Integrating edge states: " + str(t_2 - t_1) + " s")

            t_1 = time.time()
            print()
            print("=================================================")
            print()
            print("Integrating", len(self.list_of_center_state_vectors), " subset center state vectors")
            print()
            print("=================================================")
            print()
            for i, state_vector in enumerate(self.list_of_center_state_vectors):
                temp_lst.append(self._integrate_center_states(i))
            t_2 = time.time()
            print("Integrating center states: " + str(t_2 - t_1) + " s")
            self.list_of_spacecraft = temp_lst

    def _integrate_generic_states(self, i):
        if not self.do_integration:
            print("Creating ", str(i + 1) + " / " + str(len(self.list_of_state_vectors)))
            state_vector = self.list_of_state_vectors[i]
            spacecraft = sc.Spacecraft(state_vector, copy.deepcopy(self.force_model_1))
            spacecraft.integration_points = self.integration_points
            spacecraft.time_interval = self.list_of_time_ivs[i]
            spacecraft.plot_color = 'map'
        if self.do_integration:
            pause(0.05)
            print(str(i + 1) + " / " + str(len(self.list_of_state_vectors)), ": ", end="")
            spacecraft = self.list_of_spacecraft[i]
            spacecraft.integrate_states_sivp(method=self.method, rtol=self.rtol, atol=self.atol)

            # self.list_of_spacecraft.append(spacecraft)
            if not self.postintegration_check(spacecraft):
                print("Warning! Spacecraft did not pass validity check! Returning None")
                return None
            if self.direct_transformation:
                spacecraft.trajectory_conversion(mass=self.force_model_1.central_mass, transformations=self.direct_transformation)
        return spacecraft

    def _integrate_edge_states(self, i):
        state_vector = self.list_of_edge_state_vectors[i]
        spacecraft = sc.Spacecraft(state_vector, self.force_model_1)
        spacecraft.integration_points = self.integration_points
        spacecraft.time_interval = self.list_of_edge_time_ivs[i]
        spacecraft.is_edge_spacecraft = True
        spacecraft.plot_color = [0, 1, 0]
        if self.do_integration:
            pause(0.05)
            print(str(i + 1) + " / " + str(len(self.list_of_edge_state_vectors)), ": ", end="")
            spacecraft.integrate_states_sivp(method=self.method, rtol=self.rtol, atol=self.atol)

            # self.list_of_spacecraft.append(spacecraft)
            if not self.postintegration_check(spacecraft):
                print("Warning! Spacecraft did not pass validity check! Returning None")
                return None
            if self.direct_transformation:
                spacecraft.trajectory_conversion(mass=self.force_model_1.central_mass, transformations=self.direct_transformation)
        return spacecraft

    def _integrate_center_states(self, i):
        if not self.do_integration:
            state_vector = self.list_of_center_state_vectors[i]
            spacecraft = sc.Spacecraft(state_vector, self.force_model_1)
            spacecraft.integration_points = self.integration_points
            spacecraft.time_interval = self.list_of_center_time_ivs[i]
            spacecraft.is_center_spacecraft = True
            spacecraft.plot_color = [1, 0, 1]

        if self.do_integration:
            spacecraft = self.list_of_center_spacecraft[i]
            pause(0.05)
            print(str(i + 1) + " / " + str(len(self.list_of_center_state_vectors)), ": ", end="")
            spacecraft.integrate_states_sivp(method=self.method, rtol=self.rtol, atol=self.atol)

            # self.list_of_spacecraft.append(spacecraft)
            if not self.postintegration_check(spacecraft):
                print("Warning! Spacecraft did not pass validity check! Returning None")
                return None
            if self.direct_transformation:
                spacecraft.trajectory_conversion(mass=self.force_model_1.central_mass, transformations=self.direct_transformation)
        return spacecraft

    def postintegration_check(self, spacecraft):
        if len(spacecraft.trajectory_track[0]) != len(self.integration_points):
            return False
        return True

    def get_swarm_body_distances(self, body_list):
        print("Calculating body distances")
        body_trajectories = self.force_model_1.propagate_body_states(self.integration_points)
        t_1 = time.time()
        for sc in self.list_of_spacecraft:
            sc.get_body_distances(body_list=body_list, body_trajectories=body_trajectories)
        t_2 = time.time()
        print("Evaluating body distances: " + str(t_2 - t_1) + "s")
