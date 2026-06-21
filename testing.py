import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas
from numpy.f2py.crackfortran import analyzeargs

import src.spacecraft.sc
from src.system_dynamics import sd_1, SRP, atmo
from src.analysis import plotting_functions
from src.spacecraft import swarm_1
from src.astrodynamic_functions import kepler_dynamics
from src.feasibility import valid_set
from src.guidance import steering_laws
from src.computation import parallel
import math
import pickle

from src.system_dynamics.atmo import EARTH_RADIUS


def all_plots():
    t_start = 0
    t_end = 0.3e8
    integration_points = list(np.linspace(t_start, t_end, 300))
    central_mass = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/2d_dynamic.xlsx")
    force_model_1.define_central_attractor(central_mass, [0, 0, 0])

    manifolds = [[1.5e11, 1.5e11, 1],
                 [0, 0, 1],
                 [0, 0, 1],
                 [0, 0, 1],
                 [25000, 35000, 30],
                 [0, 0, 1],
                 [t_start, t_start, 1]]

    manifolds_2 = [[1.5e11, 1.5e11, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [20000, 25000, 100],
                   [0, 0, 1],
                   [t_start, t_start, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds=manifolds, force_model=force_model_1)
    sw_1.integration_points = integration_points
    sw_1.square_swarm("generic")
    sw_1.create_and_integrate_swarm(rtol=1e-3, parproc=True, cores=11)
    sw_1.get_swarm_body_distances(["body_1"])

    sw_3 = swarm_1.particle_swarm(manifolds=manifolds_2, force_model=force_model_1)
    sw_3.integration_points = integration_points
    # sw_3.square_swarm("center")
    # sw_3.square_swarm("edge")
    sw_3.square_swarm("generic")
    sw_3.create_and_integrate_swarm(rtol=1e-3, parproc=True, cores=11)
    sw_3.get_swarm_body_distances(["body_1"])

    fs = valid_set.feasibility_setup(list_of_sc=sw_3.list_of_spacecraft, force_model=force_model_1,
                                     manifolds=manifolds_2)
    fs.check_conditions()
    sw_3.list_of_spacecraft = fs.list_of_sc

    sw_2 = swarm_1.particle_swarm(manifolds=manifolds, force_model=force_model_1)
    sw_2.integration_points = integration_points
    sw_2.sensitivity_swarm(center=[1.5e11, 0, 0, 0, 35000, 0])
    sw_2.create_and_integrate_swarm(rtol=1e-3, parproc=True, cores=5)
    sw_2.get_sensitivity()

    plots = plotting_functions.graph_output(list_of_spacecraft=sw_3.list_of_spacecraft,
                                            list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=sw_1.list_of_spacecraft,
                                            force_model=force_model_1,
                                            animated=True,
                                            fps=None,
                                            axis_visibility=True,
                                            frame_delay=1)

    plots.sensitivity = sw_2.sens_mat
    plots.integration_points = integration_points

    plots.state_space_slice(index=100, slices=[["x", "vx"]])
    plots.sensitivity_plot(diag_only=True)
    plots.parameters_plot()
    plots.C3_plot()
    plots.magnitude_plot()
    plots.body_distances_plot(["body_1"])
    plots.trajectory_xyz()
    plots.moving_map_plot(match_tail_color=True)


def SSO():
    t_end = 70 * 24 * 3600  # 430 days for swarm
    sim_time = list(np.linspace(0, t_end, 10000))
    cent_mass = 5.792e24

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(mass=cent_mass, position=[0, 0, 0])
    force_model_1.central_attractor_gravity_law = kepler_dynamics.J_X_acceleration_equator_reference

    oe_vec = [15479300, 0.4, 60 * math.pi / 180, 100 * math.pi / 180, 26 * math.pi / 180, 206 * math.pi / 180]

    sc_1 = src.spacecraft.sc.Spacecraft(
        kepler_dynamics.oe_to_sv(oe_vec[0], oe_vec[1], oe_vec[2], oe_vec[3], oe_vec[4],
                                 oe_vec[5], 0, cent_mass), force_model=force_model_1)
    sc_2 = src.spacecraft.sc.Spacecraft(
        kepler_dynamics.oe_to_sv(oe_vec[0], oe_vec[1], oe_vec[2] + 20 * math.pi / 180, oe_vec[3], oe_vec[4], oe_vec[5],
                                 0, cent_mass),
        force_model=force_model_1)
    sc_3 = src.spacecraft.sc.Spacecraft(
        kepler_dynamics.oe_to_sv(oe_vec[0], oe_vec[1], oe_vec[2] + 40 * math.pi / 180, oe_vec[3], oe_vec[4], oe_vec[5],
                                 0, cent_mass),
        force_model=force_model_1)

    sc_list = [sc_1]

    setpoint = 1e-8
    for i in range(len(sc_list)):
        sc_list[i].integration_points = sim_time
        sc_list[i].time_interval = [sim_time[0], sim_time[-1]]
        sc_list[i].display_name = "Spacecraft " + str(i + 1)

        sc_list[i].integrate_states_sivp(method='DOP853', rtol=setpoint)

        sc_list[i].trajectory_to_orbital_parameters(cent_mass)
        sc_list[i].get_C3_track(central_mass=cent_mass)

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=sc_list,
                                            force_model=force_model_1, animated=False, axis_visibility=True, fps=None)

    # plt.style.use("dark_background")
    plots.parameters_plot()
    plots.trajectory_xyz()
    plots.C3_plot()
    plots.moving_map_plot()
    plt.show()
    plt.pause(1000000)
    plt.waitforbuttonpress()


def orbiting_planet():
    # Reference JD: 2453755.500000000 = A.D. 2006-Jan-20 00:00:00.0000
    t_init = 0
    t_final = 600 * 24 * 3600
    steps = 1000

    sim_time = list(np.linspace(t_init, t_final, steps))

    cent_mass = 1.98892e30

    force_model_1 = sd_1.inertial_force_model("./data/dynamic_setup.xlsx")
    force_model_1.define_central_attractor(mass=cent_mass, position=[0, 0, 0])

    manifolds = [[147000000000, 150000000000, 10],
                 [0, 0, 1],
                 [20000000000, 20000000000, 1],
                 [0, 0, 1],
                 [35955.075142841008, 35955.075142841008, 1],
                 [0, 0, 1],
                 [0, 0, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('generic')
    sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=5)
    sw_1.get_swarm_body_distances(body_list=["body_1"])
    list_of_spacecraft = sw_1.list_of_spacecraft
    list_of_spacecraft[0].plot_color = [1, 0.4, 1]

    fs = valid_set.feasibility_setup(list_of_sc=list_of_spacecraft, force_model=force_model_1, manifolds=[])
    fs.check_conditions()
    list_of_spacecraft = fs.list_of_sc

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=list_of_spacecraft,
                                            force_model=force_model_1, animated=True, axis_visibility=True, fps=None)

    # wx = output.generate_excel_output(spacecraft=list_of_spacecraft[0], inertial_force_model=force_model_1)
    # wx.write_body_trajectories()

    plots.parameters_plot()
    plots.trajectory_xyz()
    plots.C3_plot()
    plots.body_distances_plot(body_list=["body_1"])
    plots.moving_map_plot(match_tail_color=True)

    plt.show()
    plt.pause(1000000)
    plt.waitforbuttonpress()


def CR3BP():
    t_end = 30
    sim_time = list(np.linspace(0, t_end, 1000))

    force_model_1 = sd_1.CR3BP(mass_parameter=1.215058560962404E-2)
    # force_model_1 = sd_1.CR3BP(mass_parameter=3.054200000000000E-6)

    manifolds_4 = [[6.3891964038363835E-1, 6.3891964038363835E-1, 1],
                   [4.2544999999999999E-1, 4.2544999999999999E-1, 1],
                   [6.3338764777407142E-1, 6.3338764777407142E-1, 1],
                   [-3.1073088790628428E-1, -3.1073088790628428E-1, 1],
                   [-4.8651604983496066E-1, -4.8651604983496066E-1, 1],
                   [6.4806934253088289E-1, 6.4806934253088289E-1, 1],
                   [0, 0, 1]]

    manifolds_5 = [[6.3891964038363835E-1, 6.3891964038363835E-1, 1],
                   [-4.2544999999999999E-1, -4.2544999999999999E-1, 1],
                   [6.3338764777407142E-1, 6.3338764777407142E-1, 1],
                   [3.1073088790628428E-1, 3.1073088790628428E-1, 1],
                   [-4.8651604983496066E-1, -4.8651604983496066E-1, 1],
                   [-6.4806934253088289E-1, -6.4806934253088289E-1, 1],
                   [0, 0, 1]]

    manifolds_1 = [[8.6221899389004331E-1, 8.6221899389004331E-1, 1],
                   [-6.1589472580081878E-28, -6.1589472580081878E-28, 1],
                   [-9.0996038678959414E-14, -9.0996038678959414E-14, 1],
                   [1.1168790093094281E-13, 1.1168790093094281E-13, 1],
                   [8.8612113504271353E-2, 8.8612113504271353E-2, 1],
                   [-4.3863065793662631E-1, -4.3863065793662631E-1, 1],
                   [0, 0, 1]]

    manifolds_2 = [[1.1073931895066280E+0, 1.1073931895066280E+0, 1],
                   [7.1870348654174879E-23, 7.1870348654174879E-23, 1],
                   [4.5475450231117765E-14, 4.5475450231117765E-14, 1],
                   [-5.4532856949644616E-14, -5.4532856949644616E-14, 1],
                   [-2.4189177045871293E-1, -2.4189177045871293E-1, 1],
                   [4.8766778814351319E-1, 4.8766778814351319E-1, 1],
                   [0, 0, 1]]

    manifolds_3 = [[-7.9130666920915449E-1, -7.9130666920915449E-1, 1],
                   [-3.6153905370569799E-23, -3.6153905370569799E-23, 1],
                   [6.1634914945822383E-1, 6.1634914945822383E-1, 1],
                   [3.5760936793654774E-11, 3.5760936793654774E-11, 1],
                   [-2.1319406969319177E-1, -2.1319406969319177E-1, 1],
                   [6.5943284888509311E-11, 6.5943284888509311E-11, 1],
                   [0, 0, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds_1, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.direct_transformation = ("State_magnitude")
    sw_1.square_swarm('generic')
    sw_1.manifolds = manifolds_4
    sw_1.square_swarm('generic')
    sw_1.manifolds = manifolds_5
    sw_1.square_swarm('generic')
    sw_1.manifolds = manifolds_2
    sw_1.square_swarm('generic')
    sw_1.manifolds = manifolds_3
    sw_1.square_swarm('generic')
    sw_1.create_and_integrate_swarm(rtol=1e-10, parproc=True, cores=11)
    sw_1.get_swarm_body_distances(body_list=["body_1", "body_2"])

    list_of_spacecraft = sw_1.list_of_spacecraft

    list_of_spacecraft[0].plot_color = [1, 0.3, 0.3]
    list_of_spacecraft[1].plot_color = [0.3, 1, 0.5]
    list_of_spacecraft[2].plot_color = [0.3, 0.3, 1]
    list_of_spacecraft[3].plot_color = [1, 1, 0.3]
    list_of_spacecraft[4].plot_color = [1, 0.3, 1]

    """fs = valid_set.feasibility_setup(manifolds=[], list_of_sc=list_of_spacecraft, force_model=force_model_1)
    fs.check_conditions()
    list_of_spacecraft = fs.list_of_sc"""

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=list_of_spacecraft,
                                            force_model=force_model_1, animated=False, axis_visibility=True, fps=None)

    plots.state_space_slice(index=0, slices=[["vx", "vy"]])
    plots.state_space_slice(index=499, slices=[["vx", "vy"]])
    # plots.parameters_plot()
    # plots.C3_plot()
    plots.magnitude_plot()
    plots.body_distances_plot(["body_1", "body_2"])
    plots.trajectory_xyz()
    plots.moving_map_plot(plot_central_attractor=False, match_tail_color=False)


def CR3BP_ex_2():
    t_end = 20
    sim_time = list(np.linspace(0, t_end, 3))

    force_model_1 = sd_1.CR3BP(mass_parameter=1.215058560962404E-2)
    # force_model_1 = sd_1.CR3BP(mass_parameter=3.054200000000000E-6)
    steering_law = steering_laws.LocalOptimal()

    force_model_1.steering_law = steering_law
    force_model_1.get_lagrange_points()

    manifolds_4 = [[6.3891964038363835E-1, 6.3891964038363835E-1, 1],
                   [4.2544999999999999E-1, 4.2544999999999999E-1, 1],
                   [6.3338764777407142E-1, 6.3338764777407142E-1, 1],
                   [-3.1073088790628428E-1, -3.1073088790628428E-1, 1],
                   [-4.8651604983496066E-1, -4.8651604983496066E-1, 1],
                   [6.4806934253088289E-1, 6.4806934253088289E-1, 1],
                   [0, 0, 1]]

    manifolds_5 = [[6.3891964038363835E-1, 6.3891964038363835E-1, 1],
                   [-4.2544999999999999E-1, -4.2544999999999999E-1, 1],
                   [6.3338764777407142E-1, 6.3338764777407142E-1, 1],
                   [3.1073088790628428E-1, 3.1073088790628428E-1, 1],
                   [-4.8651604983496066E-1, -4.8651604983496066E-1, 1],
                   [-6.4806934253088289E-1, -6.4806934253088289E-1, 1],
                   [0, 0, 1]]

    manifolds_1 = [[8.0121899389004331E-1 - 0.1, 8.6121899389004331E-1 + 0.12, 20],
                   [-6.1589472580081878E-28, -6.1589472580081878E-28, 1],
                   [-9.0996038678959414E-14, -9.0996038678959414E-14, 1],
                   [1.1168790093094281E-13, 1.1168790093094281E-13, 1],
                   [8.2512113504271353E-2 - 0.4, 9.4712113504271353E-2 + 0.2, 20],
                   [-4.3863065793662631E-1, -4.3863065793662631E-1, 1],
                   [0, 0, 1]]

    manifolds_2 = [[1.1073931895066280E+0, 1.1073931895066280E+0, 1],
                   [7.1870348654174879E-23, 7.1870348654174879E-23, 1],
                   [4.5475450231117765E-14, 4.5475450231117765E-14, 1],
                   [-5.4532856949644616E-14, -5.4532856949644616E-14, 1],
                   [-2.4189177045871293E-1, -2.4189177045871293E-1, 1],
                   [4.8766778814351319E-1, 4.8766778814351319E-1, 1],
                   [0, 0, 1]]

    manifolds_3 = [[-7.9130666920915449E-1, -7.9130666920915449E-1, 1],
                   [-3.6153905370569799E-23, -3.6153905370569799E-23, 1],
                   [6.1634914945822383E-1, 6.1634914945822383E-1, 1],
                   [3.5760936793654774E-11, 3.5760936793654774E-11, 1],
                   [-2.1319406969319177E-1, -2.1319406969319177E-1, 1],
                   [6.5943284888509311E-11, 6.5943284888509311E-11, 1],
                   [0, 0, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds_1, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.direct_transformation = ("State_magnitude")
    sw_1.square_swarm('center')
    sw_1.create_and_integrate_swarm(method="DOP853", rtol=1e-9, parproc=True, cores=12)
    sw_1.get_swarm_body_distances(body_list=["body_1", "body_2"])

    list_of_spacecraft = sw_1.list_of_spacecraft

    fs = valid_set.feasibility_setup(manifolds=[], list_of_sc=list_of_spacecraft, force_model=force_model_1)
    fs.check_conditions()
    list_of_spacecraft = fs.list_of_sc

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=list_of_spacecraft,
                                            force_model=force_model_1, animated=False, axis_visibility=True, fps=None)

    plots.state_space_slice(index=0, slices=[["x", "vy"]])
    """
    plots.state_space_slice(index=999, slices=[["vx", "vy"], ["x", "y"]])
    plots.state_space_slice(index=999, slices=[["x", "y", "z"], ["vx", "vy", "vz"]])
    # plots.parameters_plot()
    # plots.C3_plot()
    plots.body_distances_plot(["body_1", "body_2"])"""
    # plots.trajectory_xyz()
    # plots.plot_steering()
    """plots.moving_map_plot(plot_central_attractor=False,
                          match_tail_color=False,
                          override_limits={"x": [-2, 2], "y": [-2, 2], "z": [-2, 2]},
                          k_modulo=10)"""

    plt.show()
    plt.waitforbuttonpress(10000000000)


def steering_testing():
    t_init = 0
    t_final = 70000000
    steps = 10000

    sim_time = list(np.linspace(t_init, t_final, steps))

    cent_mass = 5.972 * 10 ** 24

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(mass=cent_mass, position=[0, 0, 0])

    steering_law = steering_laws.LocalOptimal()
    steering_law.conversion_mass = cent_mass

    force_model_1.guidance = steering_law

    manifolds = [[30000000, 30000000, 1],
                 [2000000, 2000000, 1],
                 [1000000, 1000000, 1],
                 [0, 0, 1],
                 [4000, 4000, 1],
                 [1000, 1000, 1],
                 [0, 0, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('generic')
    sw_1.create_and_integrate_swarm(rtol=1e-3, parproc=False, cores=5)
    list_of_spacecraft = sw_1.list_of_spacecraft
    list_of_spacecraft[0].plot_color = [0.3, 1, 0.4]

    list_of_spacecraft[0].display_name = "Sc 1"

    force_model_1.path_to_data = "./data/target.xlsx"
    force_model_1.get_dataset()

    """# Safe the sauce with pickle
    pickle.dump(sw_1.list_of_spacecraft, open('sv.p', 'wb'))
    pickle.dump(force_model_1, open('.p', 'wb'))
    exit()

    list_of_spacecraft = pickle.load(open('sv.p', 'rb'))
    force_model_1 = pickle.load(open('.p', 'rb'))"""

    list_of_spacecraft[0].plot_color = [0.1, 1, 0.6]

    input("Start plotting? Press enter")
    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=list_of_spacecraft,
                                            force_model=force_model_1,
                                            animated=True,
                                            axis_visibility=False,
                                            fps=None)

    plots.parameters_plot()
    plots.C3_plot()
    plots.trajectory_xyz()
    plots.plot_steering_acceleration()
    plots.state_space_slice(index=100, slices=[["x", "y"]])
    plots.moving_map_plot(plot_central_attractor=True,
                          match_tail_color=False,
                          plot_planet_endpoint=False,
                          init_azim=35,
                          init_elevation=45,
                          k_modulo=15,
                          azim_rate=0.03,
                          elevation_rate=0.03)
    plt.show()
    plt.waitforbuttonpress(10000000000)


def Lagrange_targeting():
    t_init = 0
    t_final = 50000000
    steps = 10000

    sim_time = list(np.linspace(t_init, t_final, steps))

    cent_mass = 5.972 * 10 ** 24

    force_model_1 = sd_1.inertial_force_model("./data/empty_dataset.xlsx")
    force_model_1.define_central_attractor(mass=cent_mass, position=[0, 0, 0])

    steering_law = steering_laws.LocalOptimal()
    steering_law.conversion_mass = cent_mass  #  * (380073311**3) / (384748000**3)

    force_model_1.guidance = steering_law

    manifolds = [[28007331, 35007331, 20],
                 [0, 0, 1],
                 [1000, 1000, 1],
                 [0, 0, 1],
                 [3000, 3000, 1],
                 [0, 0, 1],
                 [0, 0, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds, force_model_1)
    sw_1.integration_points = sim_time
    sw_1.square_swarm('generic')
    sw_1.create_and_integrate_swarm(rtol=1e-3, parproc=True, cores=11)
    # sw_1.get_swarm_body_distances(["Moon"])
    list_of_spacecraft = sw_1.list_of_spacecraft

    input("Start plotting? Press enter")

    sc = list_of_spacecraft[0]
    long = list(np.array(sc.orbital_parameters_track[3]) + np.array(sc.orbital_parameters_track[4]) + np.array(
        sc.orbital_parameters_track[5]))
    fig = plt.figure(100)
    ax = fig.add_subplot()
    ax.plot(sim_time, long)
    plt.show()

    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=list_of_spacecraft,
                                            force_model=force_model_1,
                                            animated=False,
                                            axis_visibility=True,
                                            fps=None)

    plots.parameters_plot()
    plots.C3_plot()
    plots.trajectory_xyz()
    # plots.body_distances_plot(["Moon"])
    # plots.plot_steering()
    plots.state_space_slice(index=100, slices=[["x", "y"]])
    plots.moving_map_plot(plot_central_attractor=True,
                          match_tail_color=False,
                          plot_planet_endpoint=True,
                          init_azim=35,
                          init_elevation=90,
                          k_modulo=15)

    plt.show()
    plt.waitforbuttonpress(10000000000)


def tBP_dynamics_testing():
    t_start = 0
    t_end = 10000000
    integration_points = list(np.linspace(t_start, t_end, 10000))
    earth_mass = 5.97e24
    solar_mass = 1.989 * 10 ** 30

    force_model = sd_1.inertial_force_model(path="./data/Earth.xlsx")
    force_model.define_central_attractor(mass=solar_mass, position=[0, 0, 0])

    force_model_2 = sd_1.inertial_force_model(path="./data/empty_dataset.xlsx")
    force_model_2.define_central_attractor(mass=earth_mass, position=[0, 0, 0])

    srp_model = SRP.Solar_pressure(sail_model="ACS3", central_attractor_mass=solar_mass)
    srp_model.radiation_location = [149*10**8, 0, 0]
    srp_model.sail_control = [0.25*math.pi, 0.5*math.pi]
    # force_model.solar_pressure = srp_model

    guidance_law = steering_laws.LocalOptimal()
    guidance_law.conversion_mass = force_model.central_mass
    # force_model.guidance = guidance_law

    earth_state = kepler_dynamics.oe_to_sv(150000000000, 0, 0, 1, 1, 1, 0, force_model.central_mass)
    orbit_state = kepler_dynamics.oe_to_sv(10000000, 0, 0.1, 3, 3, 3, 0, earth_mass)
    init_state = np.array(earth_state) + np.array(orbit_state)

    sc_1 = src.spacecraft.sc.Spacecraft(init_state_vector=init_state, force_model=force_model)
    sc_1.display_name = "Full system"
    sc_1.integration_points = integration_points
    sc_1.time_interval = [t_start, t_end]
    sc_1.integrate_states_sivp(rtol=10 ** - 3)
    sc_1.trajectory_conversion(mass=solar_mass)
    sc_1.get_body_distances(body_list=["Earth"])
    sc_1.plot_color = [0.5, 1, 1]

    """fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sc_1.trajectory_track[0], sc_1.trajectory_track[1], sc_1.trajectory_track[2], color=[1, 0, 0], label="Sc")
    ax.plot(earth_traj[0], earth_traj[1], earth_traj[2], color=[0, 0, 1], label="Earth")
    ax.legend()
    plt.show()"""

    sc_2 = src.spacecraft.sc.Spacecraft(init_state_vector=orbit_state, force_model=force_model_2)
    sc_2.display_name = "Earth system"
    sc_2.integration_points = integration_points
    sc_2.time_interval = [t_start, t_end]
    sc_2.integrate_states_sivp(rtol=10 ** - 8)
    sc_2.trajectory_conversion(mass=earth_mass)
    sc_2.plot_color = [1, 0, 1]

    earth_traj = force_model.propagate_body_states(times=integration_points, mass=solar_mass, body_list=["Earth"])["Earth"]
    for i in range(6):
        sc_2.trajectory_track[i] = list(np.array(sc_2.trajectory_track[i]) + np.array(earth_traj[i]))

    list_of_sc = [sc_1, sc_2]
    sc_2.force_model = force_model
    sc_1.get_body_distances(body_list=["Earth"])
    sc_2.get_body_distances(body_list=["Earth"])
    sc_2.force_model = force_model_2

    input("Start plotting?")

    plots = plotting_functions.graph_output(list_of_spacecraft=[],
                                            list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=list_of_sc,
                                            force_model=force_model,
                                            axis_visibility=True,
                                            animated=True)

    plots.trajectory_xyz()
    plots.parameters_plot()

    sc_1.orbital_parameters_track = [[], [], [], [], [], []]
    sc_1.trajectory_to_orbital_parameters(conversion_mass=earth_mass, reference="Earth")

    plots.parameters_plot()
    plots.plot_steering_acceleration()
    plots.plot_control()
    plots.plot_target_velocity_angles()
    plots.magnitude_plot()
    plots.body_distances_plot(body_list=["Earth"])
    # plots.C3_plot()
    plots.moving_map_plot(k_modulo=10, match_tail_color=True, moving_window={"Body": "Earth", "x": 200000000, "y": 200000000, "z": 200000000}, init_azim=45, init_elevation=45)
    # plots.moving_map_plot(match_tail_color=False)
    plt.show()
    plt.waitforbuttonpress(10000000000)


def solar_pressure():
    t_start = 0
    t_end = 10**9
    integration_points = list(np.linspace(t_start, t_end, 30000))
    earth_mass = 5.97e24
    solar_mass = 1.989 * 10 ** 30

    tof_ref = [230, 460, 690, 920, 1150, 1380, 1610, 1840, 2070, 2300, 2530, 2760, 2990, 3220, 3450, 3680, 3910, 4140, 4370, 4600]
    sigma_ref = list(np.linspace(0.01, 0.20, len(tof_ref)))
    tof_lst = []
    sc_list = []
    inp = input("Load pickle? (y)")
    if inp == "y":
        force_model = pickle.load(open('.p', 'rb'))
        sc_list = pickle.load(open('sv.p', 'rb'))
    else:
        print("Integrating all initial conditions")
        for sigma in sigma_ref:

            force_model = sd_1.inertial_force_model(path="./data/empty_dataset.xlsx")
            force_model.define_central_attractor(mass=earth_mass, position=[0, 0, 0])
            force_model.central_attractor_gravity_law = src.astrodynamic_functions.kepler_dynamics.J_X_acceleration_equator_reference
            srp_model = SRP.Solar_pressure(sail_model="ACS3", central_attractor_mass=solar_mass, sigma=sigma)
            srp_model.radiation_location = [149000000000, 0, 0]
            srp_model.sail_control = [0, 0]
            force_model.solar_pressure = srp_model

            guidance_law = steering_laws.LocalOptimal()
            guidance_law.conversion_mass = earth_mass
            guidance_law.guidance_function = guidance_law.guidance_2
            guidance_law.terminator = src.guidance.steering_laws.kill_integrator_C3
            force_model.guidance = guidance_law

            orbit_state_1 = kepler_dynamics.oe_to_sv((6378+400)*1000, 0, 23.44*math.pi / 180, 3, 3, 3, 0, earth_mass)

            sc_2 = src.spacecraft.sc.Spacecraft(init_state_vector=orbit_state_1, force_model=force_model)
            sc_2.display_name = str(sigma)
            sc_2.integration_points = integration_points
            sc_2.time_interval = [t_start, t_end]
            sc_2.integrate_states_sivp(rtol=10 ** - 6)
            sc_2.trajectory_conversion(mass=5.97e24)
            sc_list.append(sc_2)

    for i, sc in enumerate(sc_list):
        print("Sigma: ", sigma_ref[i], ", ", round(sc.event_time[0][0] / (24 * 3600), 10))
        tof_lst.append(sc.event_time[0][0] / (24 * 3600))

    plt.figure()
    plt.scatter(np.array(sigma_ref) * 1000, tof_lst, color=[1, 0, 1], label="Calculated")
    plt.scatter(np.array(sigma_ref) * 1000, tof_ref, color=[0, 1, 1], label="Armando")
    plt.legend()
    plt.grid()
    plt.show()

    print("sigma_lst: ", sigma_ref)
    print("tof_lst ", tof_lst)

    inp = input("Save= (y / n)")
    if inp == "y":
        # Safe the stuff with pickle
        pickle.dump(sc_list, open('sv.p', 'wb'))
        pickle.dump(force_model, open('.p', 'wb'))

    input("Start plotting?")

    plots = plotting_functions.graph_output(list_of_spacecraft=[],
                                            list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=sc_list,
                                            force_model=force_model,
                                            axis_visibility=True,
                                            animated=False)

    plots.trajectory_xyz()
    plots.parameters_plot()
    plots.parameters_plot()
    plots.plot_steering_acceleration()
    plots.plot_control()
    plots.plot_target_velocity_angles()
    plots.magnitude_plot()
    plots.C3_plot()
    plots.moving_map_plot(k_modulo=10, match_tail_color=True)
    # plots.moving_map_plot(match_tail_color=False)
    plt.show()
    plt.waitforbuttonpress(10000000000)


def solar_swarm():
    t_start = 0
    t_end = 5000000
    integration_points = list(np.linspace(t_start, t_end, 10000))
    earth_mass = 5.97e24
    solar_mass = 1.989 * 10 ** 30

    tof_ref = [230, 460, 690, 920, 1150, 1380, 1610, 1840, 2070, 2300, 2530, 2760, 2990, 3220, 3450, 3680, 3910, 4140, 4370, 4600]
    sigma_ref = list(np.linspace(0.01, 0.20, len(tof_ref)))
    tof_lst = []

    inp = input("Load pickle? (y)")
    if inp == "y":
        force_model = pickle.load(open('.p', 'rb'))
        sc_list = pickle.load(open('sv.p', 'rb'))
    else:
        print("Integrating all initial conditions")
    sigma = 0.02

    force_model = sd_1.inertial_force_model(path="./data/empty_dataset.xlsx")
    force_model.define_central_attractor(mass=earth_mass, position=[0, 0, 0])
    # force_model.central_attractor_gravity_law = src.astrodynamic_functions.kepler_dynamics.J_X_acceleration_ecliptic_reference
    srp_model = SRP.Solar_pressure(sail_model="ACS3", central_attractor_mass=solar_mass, sigma=sigma)
    srp_model.radiation_location = [149000000000, 0, 0]
    srp_model.sail_control = [0, 0]
    force_model.solar_pressure = srp_model

    guidance_law = steering_laws.LocalOptimal()
    guidance_law.conversion_mass = earth_mass
    guidance_law.guidance_function = guidance_law.guidance_2
    guidance_law.terminator = src.guidance.steering_laws.kill_integrator_altitude
    force_model.guidance = guidance_law

    drag_model = atmo.Atmopshere()
    force_model.drag_model = drag_model

    manifolds = [[-6197696.3949212525, -6197696.3949212525, 1],
                 [2638614.7315163864, 2638614.7315163864, 1],
                 [-753362.9238320779, -753362.9238320779, 1],
                 [-3074.0790258669726, -3074.0790258669726, 1],
                 [-6384.415594809771, -6384.415594809771, 11],
                 [2928.463010243008, 2928.463010243008, 1],
                 [0, 0, 1]]

    sw_1 = swarm_1.particle_swarm(manifolds, force_model)
    sw_1.do_integration = False
    sw_1.integration_points = integration_points
    sw_1.square_swarm('generic')
    sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)
    # sw_1.get_swarm_body_distances(["Moon"])

    init_altitude = np.linspace(650000, 700000, len(sw_1.list_of_spacecraft))


    for i, sc in enumerate(sw_1.list_of_spacecraft):
        sc.force_model.solar_pressure.sail_parameters["sigma"] = 0.04
        sc.display_name = str(sc.force_model.solar_pressure.sail_parameters["sigma"]) + "   " + str(init_altitude[i]*0.001) + " km"
        sc.init_state_vector = kepler_dynamics.oe_to_sv(EARTH_RADIUS + init_altitude[i], 0, 23.44*math.pi / 180, 3, 3, 3, 0, earth_mass)

    sw_1.do_integration = True

    sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)

    for i, sc in enumerate(sw_1.list_of_spacecraft):
        try:
            print((np.linalg.norm(np.array(sc.init_state_vector[0:3])) - EARTH_RADIUS) * 0.001, "km  ", round(sc.event_time[0][0] / (24 * 3600), 10))
        except:
            pass

    """plt.figure()
    plt.scatter(np.array(sigma_ref) * 1000, tof_lst, color=[1, 0, 1], label="Calculated")
    plt.scatter(np.array(sigma_ref) * 1000, tof_ref, color=[0, 1, 1], label="Armando")
    plt.legend()
    plt.grid()
    plt.show()"""

    inp = input("Save= (y / n)")
    if inp == "y":
        # Safe the stuff with pickle
        pickle.dump(sw_1.list_of_spacecraft, open('sv.p', 'wb'))
        pickle.dump(force_model, open('.p', 'wb'))

    input("Start plotting?")

    plots = plotting_functions.graph_output(list_of_spacecraft=[],
                                            list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=sw_1.list_of_spacecraft,
                                            force_model=force_model,
                                            axis_visibility=True,
                                            animated=False)

    plots.trajectory_xyz()
    plots.parameters_plot()
    plots.plot_steering_acceleration()
    plots.plot_control()
    plots.plot_target_velocity_angles()
    plots.magnitude_plot()
    plots.plot_drag_acceleration()
    plots.C3_plot()
    plots.moving_map_plot(k_modulo=10, match_tail_color=True)
    # plots.moving_map_plot(match_tail_color=False)
    plt.show()
    plt.waitforbuttonpress(10000000000)

def atmpshere_min_altitude():
    t_start = 0
    t_end = 1000000
    integration_points = list(np.linspace(t_start, t_end, 10000))
    earth_mass = 5.97e24
    solar_mass = 1.989 * 10 ** 30

    inp = input("Load pickle? (y)")
    if inp == "y":
        force_model = pickle.load(open('.p', 'rb'))
        sc_list = pickle.load(open('sv.p', 'rb'))
    else:
        print("Integrating all initial conditions")
    sigma = 0.02

    force_model = sd_1.inertial_force_model(path="./data/empty_dataset.xlsx")
    force_model.define_central_attractor(mass=earth_mass, position=[0, 0, 0])
    # force_model.central_attractor_gravity_law = src.astrodynamic_functions.kepler_dynamics.J_X_acceleration_ecliptic_reference
    srp_model = SRP.Solar_pressure(sail_model="ACS3", central_attractor_mass=solar_mass, sigma=sigma)
    srp_model.radiation_location = [149000000000, 0, 0]
    srp_model.sail_control = [0, 0]
    force_model.solar_pressure = srp_model

    guidance_law = steering_laws.LocalOptimal()
    guidance_law.conversion_mass = earth_mass
    guidance_law.guidance_function = guidance_law.guidance_2
    guidance_law.terminator = src.guidance.steering_laws.kill_integrator_altitude
    force_model.guidance = guidance_law

    drag_model = atmo.Atmopshere()
    force_model.drag_model = drag_model

    init_ang = np.linspace(0, 2*math.pi, 20)
    sail_loading = np.linspace(0.02, 0.2, 20)

    min_alt = []

    for i in range(len(init_ang)):
        for j in range(len(sail_loading)):

            manifolds = [[-6197696.3949212525, -6197696.3949212525, 1],
                         [2638614.7315163864, 2638614.7315163864, 1],
                         [-753362.9238320779, -753362.9238320779, 1],
                         [-3074.0790258669726, -3074.0790258669726, 1],
                         [-6384.415594809771, -6384.415594809771, 11],
                         [2928.463010243008, 2928.463010243008, 1],
                         [0, 0, 1]]

            sw_1 = swarm_1.particle_swarm(manifolds, force_model)
            sw_1.do_integration = False
            sw_1.integration_points = integration_points
            sw_1.square_swarm('generic')
            sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)

            sample_altitude = np.linspace(600000, 770000, manifolds[4][2])

            for k, sc in enumerate(sw_1.list_of_spacecraft):
                sc.force_model.solar_pressure.sail_parameters["sigma"] = sail_loading[j]
                sc.init_state_vector = kepler_dynamics.oe_to_sv(EARTH_RADIUS + sample_altitude[k], 0, 23.44*math.pi / 180, 3, 3, init_ang[i], 0, earth_mass)

            sw_1.do_integration = True

            sw_1.create_and_integrate_swarm(rtol=1e-6, parproc=True, cores=11)

            altitude_list = [(np.linalg.norm(np.array(sc.init_state_vector[0:3])) - EARTH_RADIUS) * 0.001 for sc in sw_1.list_of_spacecraft]
            altitude_list = sorted(altitude_list)

            event_list = []
            for sc in sw_1.list_of_spacecraft:
                if sc.event_time:
                    event_list.append(sc.event_time)
            event_list = sorted(event_list)
            l = len(event_list)

            try:
                if l == 0:
                    min_alt.append({"TAEPO": round(float(init_ang[i]) * 180 / math.pi, 3), "Sigma": round(float(sail_loading[j]), 3), "Cutoff": round(altitude_list[l], 3)})
                else:
                    cutoff_alt_1 = altitude_list[l]
                    cutoff_alt_2 = altitude_list[l-1]
                    min_alt.append({"TAEPO": round(float(init_ang[i]) * 180 / math.pi, 3), "Sigma": round(float(sail_loading[j]), 3), "Cutoff": round((cutoff_alt_1 + cutoff_alt_2) / 2, 3)})
            except:
                min_alt.append({"TAEPO": round(float(init_ang[i]) * 180 / math.pi, 3), "Sigma": round(float(sail_loading[j]), 3), "Cutoff": None})

            print(min_alt[-1])

    sigma_lst = []
    ang_lst = []
    cutoff_lst = []

    for alt in min_alt:
        sigma_lst.append(alt["Sigma"])
        ang_lst.append(alt["TAEPO"])
        cutoff_lst.append(alt["Cutoff"])

    out_df = pandas.DataFrame()
    out_df["Sigma"] = sigma_lst
    out_df["TAEPO"] = ang_lst
    out_df["Cutoff"] = cutoff_lst
    out_df.to_excel("Altitude_cutoff_data.xlsx")

    """input("Start plotting?")

    plots = plotting_functions.graph_output(list_of_spacecraft=[],
                                            list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=sw_1.list_of_spacecraft,
                                            force_model=force_model,
                                            axis_visibility=True,
                                            animated=False)

    plots.trajectory_xyz()
    plots.parameters_plot()
    plots.plot_steering_acceleration()
    plots.plot_control()
    plots.plot_target_velocity_angles()
    plots.magnitude_plot()
    plots.plot_drag_acceleration()
    plots.C3_plot()
    plots.moving_map_plot(k_modulo=10, match_tail_color=True)
    # plots.moving_map_plot(match_tail_color=False)
    plt.show()
    plt.waitforbuttonpress(10000000000)"""


# ex_7_SSO()
if __name__ == "__main__":
    # CR3BP()
    # CR3BP_ex_2()
    # SSO()
    # steering_testing()
    # orbiting_planet()
    # Lagrange_targeting()
    atmpshere_min_altitude()
