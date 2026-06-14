import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import src.spacecraft.sc
from src.system_dynamics import sd_1
from src.analysis import plotting_functions, output
from src.spacecraft import swarm_1, sc
from src.astrodynamic_functions import kepler_dynamics
from src.feasibility import valid_set


def ex_8():
    # Voyager propagation
    t_start = 2 * 24 * 3600
    t_end = 5e8
    sim_time = list(np.linspace(t_start, t_end, 2000))
    central_attractor = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/solar_system_flat_1977_08_20_00_00_00.xlsx")
    force_model_1.define_central_attractor(central_attractor, [0, 0, 0])

    manifolds = [[0.8e10, 1.2e10, 50],
                 [-5.3e10, -5.1e10, 50],
                 [0, 0, 1],
                 [60000, 60000, 1],
                 [0, 0, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    Voyager_2 = [[1.301321370230550E+11, 1.301321370230550E+11, 1],
                 [-7.657362855080509E+10, -7.657362855080509E+10, 1],
                 [0 * 3.855886646107025E+08, 0 * 3.855886646107025E+08, 1],
                 [1.71707E+04 + 20, 1.71707E+04 + 60, 10],
                 [3.4783853E+04 - 60, 3.4783853E+04, 10],
                 [0 * 3.255E+03, 0 * 3.295E+03, 1],
                 [t_start, t_start, 1]]

    Voyager_2_2 = [[1.301321370230550E+11, 1.301321370230550E+11, 1],
                   [-7.657362855080509E+10, -7.657362855080509E+10, 1],
                   [0 * 3.855886646107025E+08, 0 * 3.855886646107025E+08, 1],
                   [1.72046078838E+04-0.03, 1.72046078838E+04+0.03, 10],
                   [3.4738504E+04-0.03, 3.4738504E+04+0.03, 10],
                   [0 * 3.255E+03, 0 * 3.295E+03, 1],
                   [t_start, t_start, 1]]

    sw_1 = swarm_1.particle_swarm(Voyager_2_2, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('edge')
    sw_1.square_swarm("center")
    sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)

    sw_2 = swarm_1.particle_swarm([], force_model_1)
    sw_2.integration_points = sim_time
    sw_2.sensitivity_swarm(
        center=[1.301321370230550E+11, -7.657362855080509E+10, 0 * 3.855886646107025E+08, 1.72046078838E+04, 3.4738504E+04,
                0 * 3.265482868525462E+03])
    sw_2.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)
    sw_2.get_sensitivity()

    sw_1.get_swarm_body_distances(["Jupiter", "Saturn", "Uranus", "Neptune"])

    fs = valid_set.feasibility_setup(sw_1.list_of_spacecraft, force_model_1, manifolds)
    fs.check_conditions()

    NH_special = [sw_1.list_of_spacecraft[0]]
    NH_special[0].plot_color = [1, 0, 0]
    NH_special[0].is_center_spacecraft = True

    # list_of_spacecraft=pickle.load(open('sv.p','rb'))

    # Safe the stuff with pickle
    # pickle.dump(list_of_spacecraft, open('sv.p', 'wb'))
    # pickle.dump(force_model_1, open('.p', 'wb'))

    # Do the condition checking
    """fs = valid_set.feasibility_setup(list_of_spacecraft, force_model_1, manifolds)
    fs.origin = np.array([0, 0, 0, 0, 0, 0])
    fs.normal = np.array([0, 0, 0, -1, 0, 0])
    fs.sample_points = [299]
    fs.return_type = "all"
    fs.fill_num = 2
    fs.convex_fill_points = 100
    fs.resample_edge_only = False
    fs.cutoff_dist = 5000
    fs.check_planar_conditions()
    fs.create_resampled_sc()
    list_of_spacecraft = fs.list_of_sc + fs.list_of_res_sc
    list_of_resampled_spacecraft = fs.list_of_res_sc"""

    # sv_safe = kepler_dynamics.get_sv_list(list_of_resampled_spacecraft, 99)

    # Safe the stuff with pickle
    # pickle.dump(sv_safe, open('sv.p', 'wb'))
    # pickle.dump(force_model_1, open('.p', 'wb'))

    input("Start plotting?")
    plots = plotting_functions.graph_output(sw_1.list_of_spacecraft, [], [], force_model_1,
                                            animated=True, frame_delay=1)
    plots.slices = [["x", "y"], ["vx", "vy"], ["x", "vx", "vy"]]
    plots.slice_index = 99
    plots.body_plots = ["Jupiter", "Pluto"]
    plots.sensitivity = sw_2.sens_mat

    plots.state_space_slice(index=0, slices=[["vx", "vy"]])

    plots.C3_plot()
    # plots.parameters_plot()
    # plots.sensitivity_plot(diag_only=False)
    # plots.trajectory_xyz()
    plots.moving_map_plot()

def ex_9():
    # 2d initial region set boundary intersection
    t_start = 0
    t_end = 1.5e5
    sim_time = list(np.linspace(t_start, t_end, 300))
    central_attractor = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(central_attractor, [0, 0, 0])

    manifolds = [[1e10, 1.05e10, 700],
                 [0, 1.05e9, 700],
                 [0, 0, 1],
                 [0, 0, 1],
                 [50000, 50000, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    manifolds_2 = [[1e10, 1.05e10, 20],
                 [0, 1.05e9, 20],
                 [0, 0, 1],
                 [0, 0, 1],
                 [50000, 50000, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('edge')
    # sw_1.manifolds = manifolds_2
    # sw_1.square_swarm("center")
    sw_1.create_and_integrate_swarm(rtol=1e-9, parproc=True, cores=11)

    list_of_spacecraft = sw_1.list_of_spacecraft
    # Do the condition checking
    fs = valid_set.feasibility_setup(list_of_spacecraft, force_model_1, manifolds)
    fs.origin = np.array([0.5e10, 0, 0, 0, 0, 0])
    fs.normal = np.array([1, 0, 0, 0, 0, 0])
    fs.sample_points = [299]
    fs.return_type = "all"
    fs.fill_num = 2
    fs.convex_fill_points = 500
    fs.resample_edge_only = True
    fs.cutoff_dist = 5e6
    fs.check_planar_conditions()
    fs.create_resampled_sc()
    list_of_spacecraft = fs.list_of_sc
    # list_of_resampled_spacecraft = fs.list_of_res_sc

    # sv_safe = kepler_dynamics.get_sv_list(list_of_resampled_spacecraft, 99)

    # Safe the stuff with pickle
    # pickle.dump(sv_safe, open('sv.p', 'wb'))
    # pickle.dump(force_model_1, open('.p', 'wb'))

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft, fs.list_of_res_sc, [], force_model_1,
                                            animated=True, frame_delay=1)
    plots.slices = [["x", "y"], ["vx", "vy"], ["x", "vx", "vy"]]

    plots.state_space_slice(index=0, slices=[["vx", "vy"]])
    plots.state_space_slice(index=299, slices=[["vx", "vy"]])
    plots.state_space_slice(index=0, slices=[["x", "y"]])
    plots.state_space_slice(index=299, slices=[["x", "y"]])

    plots.trajectory_xyz()
    plots.moving_map_plot()


def ex_11():
    t_start = 0
    t_end = 1.35e5
    sim_time = list(np.linspace(t_start, t_end, 300))
    central_attractor = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(central_attractor, [0, 0, 0])

    manifolds = [[1e10, 1.05e10, 20],
                 [1e10, 1.05e10, 20],
                 [0, 0, 1],
                 [0, 0, 1],
                 [-50000, -50000, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    manifolds_2 = [[-1e10, -1.05e10, 20],
                 [1e10, 1.05e10, 20],
                 [0, 0, 1],
                 [0, 0, 1],
                 [-50000, -50000, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]


    sw_1 = swarm_1.particle_swarm(manifolds, force_model_1)
    sw_1.integration_points = sim_time
    # sw_1.square_swarm('edge')
    sw_1.square_swarm("center")
    sw_1.manifolds = manifolds_2
    # sw_1.square_swarm('edge')
    sw_1.square_swarm("center")
    sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)
    list_of_spacecraft = sw_1.list_of_spacecraft

    fs = valid_set.feasibility_setup(list_of_sc=list_of_spacecraft, force_model=force_model_1, manifolds=manifolds)
    fs.check_conditions()
    list_of_spacecraft = fs.list_of_sc

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft, [], [], force_model_1,
                                            animated=False, frame_delay=1)

    plots.state_space_slice(index=0, slices=[["vx", "vy"]])
    plots.state_space_slice(index=299, slices=[["vx", "vy"]])
    plots.state_space_slice(index=0, slices=[["x", "y"]])
    plots.state_space_slice(index=299, slices=[["x", "y"]])

    plots.state_space_slice(index=299, slices=[["vx", "vy", "x"]])
    plots.state_space_slice(index=299, slices=[["vx", "vy", "y"]])

    plots.C3_plot()

    plots.trajectory_xyz()
    plots.moving_map_plot()

def integrator_comparison():
    t_end = 8e7
    sim_time = list(np.linspace(0, t_end, 1000))
    cent_mass = 1e30

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(mass=cent_mass, position=[0, 0, 0])

    state_vector = [1e11, 0, 0, 0, 15000, 0]
    sc_1 = src.spacecraft.sc.Spacecraft(state_vector, force_model=force_model_1)
    sc_2 = src.spacecraft.sc.Spacecraft(state_vector, force_model=force_model_1)
    sc_3 = src.spacecraft.sc.Spacecraft(state_vector, force_model=force_model_1)
    sc_4 = src.spacecraft.sc.Spacecraft(state_vector, force_model=force_model_1)
    sc_5 = src.spacecraft.sc.Spacecraft(state_vector, force_model=force_model_1)
    sc_6 = src.spacecraft.sc.Spacecraft(state_vector, force_model=force_model_1)

    sc_1.display_name = 'RK23'
    sc_2.display_name = 'RK45'
    sc_3.display_name = 'DOP853'
    sc_4.display_name = 'Radau'
    sc_5.display_name = 'BDF'
    sc_6.display_name = 'LSODA'

    sc_list = [sc_1, sc_2, sc_3, sc_4, sc_5, sc_6]
    for i in range(len(sc_list)):
        sc_list[i].integration_points = sim_time
        sc_list[i].time_interval = [sim_time[0], sim_time[-1]]

    setpoint = 1e-3
    sc_1.integrate_states_sivp(method='RK23', rtol=setpoint)
    sc_2.integrate_states_sivp(method='RK45', rtol=setpoint)
    sc_3.integrate_states_sivp(method='DOP853', rtol=setpoint)
    sc_4.integrate_states_sivp(method='Radau', rtol=setpoint)
    sc_5.integrate_states_sivp(method='BDF', rtol=setpoint)
    sc_6.integrate_states_sivp(method='LSODA', rtol=setpoint)

    for i in range(len(sc_list)):
        sc_list[i].trajectory_to_orbital_parameters(cent_mass)
        sc_list[i].get_C3_track(central_mass=cent_mass)

    sc_1.plot_color = [1, 0, 0]
    sc_2.plot_color = [0, 1, 0]
    sc_3.plot_color = [0, 0, 1]
    sc_4.plot_color = [1, 0, 1]
    sc_5.plot_color = [1, 1, 0]
    sc_6.plot_color = [0, 1, 1]

    for sc in sc_list:
        energy_change = sc.C3_track[-1] + 1.10986e9
        rel = energy_change / 1.10986e9
        print(sc.display_name, " ", rel)

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=sc_list,
                                            force_model=force_model_1, animated=True)

    plots.parameters_plot()
    plots.trajectory_xyz()
    plots.C3_plot()
    plots.moving_map_plot()
    plt.show()
    plt.pause(1000000)
    plt.waitforbuttonpress()


def Voyager_2():
    t_start = 2 * 24 * 3600
    t_end = 5e8
    sim_time = list(np.linspace(t_start, t_end, 1000))
    central_attractor = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/solar_system_flat_1977_08_20_00_00_00.xlsx")
    force_model_1.define_central_attractor(central_attractor, [0, 0, 0])

    manifolds = [[0.8e10, 1.2e10, 50],
                 [-5.3e10, -5.1e10, 50],
                 [0, 0, 1],
                 [60000, 60000, 1],
                 [0, 0, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    Voyager_2_orig = [[1.301321370230550E+11, 1.301321370230550E+11, 1],
                      [-7.657362855080509E+10, -7.657362855080509E+10, 1],
                      [0 * 3.855886646107025E+08, 0 * 3.855886646107025E+08, 1],
                      [1.72046078838E+04 - 0.02, 1.72046078838E+04 + 0.02, 30],
                      [3.4738504E+04 - 0.02, 3.4738504E+04 + 0.02, 30],
                      [0 * 3.255E+03, 0 * 3.295E+03, 1],
                      [t_start, t_start, 1]]

    Voyager_2_all = [[1.301321370230550E+11, 1.301321370230550E+11, 1],
                     [-7.657362855080509E+10, -7.657362855080509E+10, 1],
                     [0 * 3.855886646107025E+08, 0 * 3.855886646107025E+08, 1],
                     [1.72046078838E+04 - 0.02, 1.72046078838E+04 + 0.02, 40],
                     [3.4738504E+04 - 0.02, 3.4738504E+04 + 0.02, 40],
                     [0 * 3.255E+03, 0 * 3.295E+03, 1],
                     [t_start, t_start, 1]]

    Voyager_2_2_Jupiter_target = [[1.301321370230550E+11, 1.301321370230550E+11, 1],
                                  [-7.657362855080509E+10, -7.657362855080509E+10, 1],
                                  [0 * 3.855886646107025E+08, 0 * 3.855886646107025E+08, 1],
                                  [1.72046078838E+04 - 15, 1.72046078838E+04 + 5, 40],
                                  [3.4738504E+04 - 5, 3.4738504E+04 + 15, 40],
                                  [0 * 3.255E+03, 0 * 3.295E+03, 1],
                                  [t_start, t_start, 1]]

    """sw_1 = swarm_1.particle_swarm(Voyager_2_all, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('edge')
    sw_1.square_swarm("center")
    sw_1.integrate_swarm(rtol=1e-6, parproc=True, cores=11)

    sw_1.get_swarm_body_distances(["Jupiter", "Saturn", "Uranus", "Neptune"])

    # Safe the sauce with pickle
    pickle.dump(sw_1.list_of_spacecraft, open('sv.p', 'wb'))
    pickle.dump(force_model_1, open('.p', 'wb'))
    exit()"""

    sw_1 = swarm_1.particle_swarm(manifolds=[], force_model=force_model_1)
    sw_1.integration_points = sim_time
    sw_1.sensitivity_swarm(
        center=[1.301321370230550E+11, -7.657362855080509E+10, 0, 1.7204E+04 + 0.61, 3.4738504E+04 + 0.49, 0])
    sw_1.create_and_integrate_swarm(rtol=1e-6)
    sw_1.get_sensitivity()

    list_of_spacecraft = pickle.load(open('sv.p', 'rb'))
    force_model_1 = pickle.load(open('.p', 'rb'))

    # Remember to modify the conditions accordingly!
    fs = valid_set.feasibility_setup(list_of_spacecraft, force_model_1, manifolds)
    fs.check_conditions()

    NH_special = [list_of_spacecraft[0]]
    NH_special[0].plot_color = [1, 0, 0]
    NH_special[0].is_center_spacecraft = True

    input("Start plotting?")
    plots = plotting_functions.graph_output(fs.list_of_sc, [], [], force_model_1,
                                            animated=True, frame_delay=1, fps=None, axis_visibility=True)

    plots.sensitivity = sw_1.sens_mat

    plots.state_space_slice(index=0, slices=[["vx", "vy"]])

    plots.C3_plot()
    plots.body_distances_plot(["Jupiter", "Saturn", "Uranus", "Neptune"])
    plots.sensitivity_plot(diag_only=True)
    plots.sensitivity_plot(diag_only=False)
    plots.moving_map_plot()


def ex_9():
    t_start = 0
    t_end = 1.5e5
    sim_time = list(np.linspace(t_start, t_end, 300))
    central_attractor = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(central_attractor, [0, 0, 0])

    manifolds = [[1.03e10, 1.04e10, 500],
                 [0, 0.6e9, 500],
                 [0, 0, 1],
                 [50000, 50000, 1],
                 [0, 0, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    manifolds_2 = [[1.03e10, 1.04e10, 10],
                   [0, 0.6e9, 10],
                   [0, 0, 1],
                   [50000, 50000, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [t_start, t_start, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('edge')
    sw_1.manifolds = manifolds_2
    sw_1.square_swarm("center")
    sw_1.create_and_integrate_swarm(rtol=1e-12, parproc=True, cores=11)

    list_of_spacecraft = sw_1.list_of_spacecraft
    # Do the condition checking
    fs = valid_set.feasibility_setup(list_of_spacecraft, force_model_1, manifolds)
    fs.origin = np.array([5e9, 0, 0, 0, 0, 0])
    fs.normal = np.array([1, 0, 0, 0, 0, 0])
    fs.sample_points = [299]
    fs.return_type = "base"
    fs.fill_num = 2
    fs.convex_fill_points = 500
    fs.resample_edge_only = True
    fs.cutoff_dist = 1e8
    fs.check_planar_conditions(naive_sampling=False, full_projection=False, discretization=20)
    fs.create_resampled_sc()
    list_of_spacecraft = fs.list_of_sc
    # list_of_resampled_spacecraft = fs.list_of_res_sc

    # sv_safe = kepler_dynamics.get_sv_list(list_of_resampled_spacecraft, 99)

    # Safe the stuff with pickle
    pickle.dump(list_of_spacecraft, open('sv.p', 'wb'))
    pickle.dump(force_model_1, open('.p', 'wb'))

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft, fs.list_of_res_sc, [], force_model_1,
                                            animated=False, frame_delay=1)
    plots.slices = [["x", "y"], ["vx", "vy"], ["x", "vx", "vy"]]

    plots.state_space_slice(index=0, slices=[["vx", "vy"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=299, slices=[["vx", "vy"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=0, slices=[["x", "y"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=299, slices=[["x", "y"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=0, slices=[["x", "y", "z"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=299, slices=[["x", "y", "z"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=0, slices=[["x", "y", "vx"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=299, slices=[["x", "y", "vx"]], edge=False, center=True, resample=True)

    plots.trajectory_xyz()
    plots.moving_map_plot()


def set_intersection_counterexample():
    t_start = 0
    t_end = 0.5e5
    sim_time = list(np.linspace(t_start, t_end, 300))
    central_attractor = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(central_attractor, [0, 0, 0])

    manifolds = [[0.5e10, 0.55e10, 6],
                 [5e9, 5.5e9, 6],
                 [0, 0, 1],
                 [0, 0, 1],
                 [-70000, -70000, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    manifolds_2 = [[-0.5e10, -0.55e10, 6],
                   [5e9, 5.5e9, 6],
                   [0, 0, 1],
                   [0, 0, 1],
                   [-70000, -70000, 1],
                   [0, 0, 1],
                   [t_start, t_start, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm("center")
    sw_1.manifolds = manifolds_2
    sw_1.square_swarm("center")
    sw_1.create_and_integrate_swarm(rtol=1e-3, parproc=True, cores=11)
    list_of_spacecraft = sw_1.list_of_spacecraft

    fs = valid_set.feasibility_setup(list_of_sc=list_of_spacecraft, force_model=force_model_1, manifolds=manifolds)
    fs.check_conditions()
    list_of_spacecraft = fs.list_of_sc

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft, [], [], force_model_1,
                                            animated=True, frame_delay=1, fps=None)
    plots.slices = [["x", "y"], ["vx", "vy"], ["x", "vx", "vy"]]

    plots.state_space_slice(index=299, slices=[["vx", "vy"]])
    plots.state_space_slice(index=299, slices=[["x", "y"]])
    plots.state_space_slice(index=299, slices=[["x", "vx"]])
    plots.state_space_slice(index=299, slices=[["y", "vy"]])
    plots.state_space_slice(index=299, slices=[["vx", "vy", "x"]])
    plots.state_space_slice(index=299, slices=[["vx", "vy", "y"]])

    plots.trajectory_xyz()
    plots.moving_map_plot()


def ex_11():
    t_start = 0
    t_end = 1
    sim_time = list(np.linspace(t_start, t_end, 500))
    central_attractor = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(central_attractor, [0, 0, 0])

    manifolds = [[1e10, 1.05e10, 15],
                 [1e10, 1.05e10, 15],
                 [0, 5e8, 15],
                 [0, 0, 1],
                 [-50000, -50000, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    manifolds_2 = [[-1e10, -1.05e10, 20],
                   [1e10, 1.05e10, 20],
                   [0, 0, 1],
                   [0, 0, 1],
                   [-50000, -50000, 1],
                   [0, 0, 1],
                   [t_start, t_start, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds_2, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('edge')
    sw_1.square_swarm("center")
    sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)
    list_of_spacecraft = sw_1.list_of_spacecraft

    fs = valid_set.feasibility_setup(list_of_sc=list_of_spacecraft, force_model=force_model_1, manifolds=manifolds)
    fs.check_conditions()
    list_of_spacecraft = fs.list_of_sc

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft, [], [], force_model_1,
                                            animated=False, frame_delay=1, fps=None, axis_visibility=True)

    plots.state_space_slice(index=0, slices=[["x", "y", "z"]])

    plots.C3_plot()

    plots.trajectory_xyz()
    plots.moving_map_plot()


def dynamics_example_2d():
    t_end = 1000 * 24 * 3600  # 430 days for swarm
    sim_time = list(np.linspace(0, t_end, 500))
    cent_mass = 1e30

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(mass=cent_mass, position=[0, 0, 0])

    """state_vector = [1e11, 0, 0, 0, 14540, 0]

    sc_ref = src.spacecraft.sc.Spacecraft(state_vector, inertial_force_model=force_model_1)

    sc_ref.display_name = "Test mass"


    sc_list = [sc_ref]
    for i in range(len(sc_list)):
        sc_list[i].integration_points = sim_time
        sc_list[i].time_interval = [sim_time[0], sim_time[-1]]

    setpoint = 1e-8
    sc_list[0].integrate_states_sivp(method='DOP853', rtol=setpoint)

    for i in range(len(sc_list)):
        sc_list[i].trajectory_to_orbital_parameters(cent_mass)
        sc_list[i].get_C3_track(central_mass=cent_mass)

    sc_ref.plot_color = [0.7, 0, 0.7]"""

    manifolds_2 = [[1.2e11, 1.21e11, 1],
                   [0, 1e10, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [14540, 14540, 1],
                   [1000, 1000, 1],
                   [0, 0, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds_2, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('generic')
    sw_1.create_and_integrate_swarm(rtol=1e-3, parproc=True, cores=5)
    # sw_1.get_swarm_body_distances(body_list=["body_1", "body_2", "body_3", "body_4"])
    list_of_spacecraft = sw_1.list_of_spacecraft
    list_of_spacecraft[0].plot_color = [1, 1, 1]

    """fs = valid_set.feasibility_setup(manifolds=[], list_of_sc=list_of_spacecraft, inertial_force_model=force_model_1)
    fs.check_conditions()
    list_of_spacecraft = fs.list_of_sc"""

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=list_of_spacecraft,
                                            force_model=force_model_1, animated=True, axis_visibility=True, fps=None)

    plots.parameters_plot()
    plots.trajectory_xyz()
    plots.C3_plot()
    # plots.body_distances_plot(body_list=["body_1", "body_2", "body_3", "body_4"])
    plots.magnitude_plot()
    plots.state_space_slice(index=0, slices=[["x", "y"]])
    plots.state_space_slice(index=1, slices=[["x", "y", "z"]])
    plots.moving_map_plot()
    plt.show()
    plt.pause(1000000)
    plt.waitforbuttonpress()


def ex_13():
    t_start = 0
    t_end = 1.3e5
    sim_time = list(np.linspace(t_start, t_end, 300))
    central_attractor = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(central_attractor, [0, 0, 0])

    manifolds = [[1.3e10, 1.4e10, 20],
                 [0, 6e9, 20],
                 [0, 6e9, 20],
                 [50000, 50000, 1],
                 [0, 0, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    manifolds_2 = [[1.3e10, 1.4e10, 5],
                   [0, 6e9, 5],
                   [0, 6e9, 5],
                   [50000, 50000, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [t_start, t_start, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('edge')
    sw_1.manifolds = manifolds_2
    sw_1.square_swarm("center")
    sw_1.create_and_integrate_swarm(rtol=1e-12, parproc=True, cores=11)

    list_of_spacecraft = sw_1.list_of_spacecraft
    # Do the condition checking
    fs = valid_set.feasibility_setup(list_of_spacecraft, force_model_1, manifolds)
    fs.origin = np.array([1.35e10, 0, 0, 0, 0, 0])
    fs.normal = np.array([1, 0, 0, 0, 0, 0])
    fs.sample_points = [299]
    fs.return_type = "base"
    fs.fill_num = 2
    fs.convex_fill_points = 500
    fs.resample_edge_only = True
    fs.cutoff_dist = 1e10
    fs.check_planar_conditions(naive_sampling=True, full_projection=True, discretization=20)
    fs.create_resampled_sc()
    list_of_spacecraft = fs.list_of_sc
    # list_of_resampled_spacecraft = fs.list_of_res_sc

    # sv_safe = kepler_dynamics.get_sv_list(list_of_resampled_spacecraft, 99)

    # Safe the stuff with pickle
    # pickle.dump(list_of_spacecraft, open('sv.p', 'wb'))
    # pickle.dump(force_model_1, open('.p', 'wb'))

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft, fs.list_of_res_sc, [], force_model_1,
                                            animated=False, frame_delay=1)
    plots.slices = [["x", "y"], ["vx", "vy"], ["x", "vx", "vy"]]

    plots.state_space_slice(index=0, slices=[["vx", "vy"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=299, slices=[["vx", "vy"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=0, slices=[["x", "y"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=299, slices=[["x", "y"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=0, slices=[["x", "y", "z"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=299, slices=[["x", "y", "z"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=0, slices=[["x", "y", "vx"]], edge=False, center=True, resample=True)
    plots.state_space_slice(index=299, slices=[["x", "y", "vx"]], edge=False, center=True, resample=True)

    plots.trajectory_xyz()
    plots.moving_map_plot()


def boundary_feasibility_base():
    # 2d initial region set boundary intersection
    t_start = 0
    t_end = 1.5e5
    sim_time = list(np.linspace(t_start, t_end, 300))
    central_attractor = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(central_attractor, [0, 0, 0])

    manifolds = [[1e10, 1.05e10, 700],
                 [0, 1.05e9, 700],
                 [0, 0, 1],
                 [0, 0, 1],
                 [50000, 50000, 1],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    manifolds_2 = [[1e10, 1.05e10, 50],
                   [0, 1.05e9, 50],
                   [0, 0, 1],
                   [0, 0, 1],
                   [50000, 50000, 1],
                   [0, 0, 1],
                   [t_start, t_start, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('edge')
    sw_1.manifolds = manifolds_2
    # sw_1.square_swarm("center")
    sw_1.create_and_integrate_swarm(rtol=1e-9, parproc=True, cores=5)

    list_of_spacecraft = sw_1.list_of_spacecraft
    # Do the condition checking
    fs = valid_set.feasibility_setup(list_of_spacecraft, force_model_1, manifolds)
    fs.origin = np.array([0.5e10, 0, 0, 0, 0, 0])
    fs.normal = np.array([1, 0, 0, 0, 0, 0])
    fs.sample_points = [299]
    fs.return_type = "all"
    fs.fill_num = 2
    fs.convex_fill_points = 500
    fs.resample_edge_only = True
    fs.cutoff_dist = 5e6
    fs.check_planar_conditions(naive_sampling=False, full_projection=True, discretization=50)
    list_of_spacecraft = fs.list_of_sc
    fs.create_resampled_sc()
    # list_of_spacecraft = fs.list_of_sc

    # sv_safe = kepler_dynamics.get_sv_list(list_of_resampled_spacecraft, 99)

    # Safe the stuff with pickle
    # pickle.dump(sv_safe, open('sv.p', 'wb'))
    # pickle.dump(force_model_1, open('.p', 'wb'))

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft, fs.list_of_res_sc, [], force_model_1,
                                            animated=True, frame_delay=1, fps=None, axis_visibility=True)
    plots.slices = [["x", "y"], ["vx", "vy"], ["x", "vx", "vy"]]

    plots.state_space_slice(index=0, slices=[["vx", "vy"]])
    plots.state_space_slice(index=299, slices=[["vx", "vy"]])
    plots.state_space_slice(index=0, slices=[["x", "y"]])
    plots.state_space_slice(index=299, slices=[["x", "y"]])
    plots.state_space_slice(index=299, slices=[["x", "y", "vx"]])
    plots.state_space_slice(index=299, slices=[["x", "y", "vy"]])
    plots.state_space_slice(index=299, slices=[["x", "vy", "vx"]])
    plots.state_space_slice(index=299, slices=[["y", "vy", "vx"]])

    plots.trajectory_xyz()
    plots.moving_map_plot()