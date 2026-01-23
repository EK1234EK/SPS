import numpy as np
import pickle
import matplotlib.pyplot as plt
import src.spacecraft.sc
from src.system_dynamics import sd_1
from src.analysis import plotting_functions
from src.spacecraft import swarm_1
from src.astrodynamic_functions import kepler_dynamics
from src.feasibility import valid_set
import math


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
    sw_1.integrate_swarm(rtol=1e-3, parproc=True, cores=11)
    sw_1.get_swarm_body_distances(["body_1"])

    sw_3 = swarm_1.particle_swarm(manifolds=manifolds_2, force_model=force_model_1)
    sw_3.integration_points = integration_points
    # sw_3.square_swarm("center")
    # sw_3.square_swarm("edge")
    sw_3.square_swarm("generic")
    sw_3.integrate_swarm(rtol=1e-3, parproc=True, cores=11)
    sw_3.get_swarm_body_distances(["body_1"])

    fs = valid_set.feasibility_setup(list_of_sc=sw_3.list_of_spacecraft, force_model=force_model_1,
                                     manifolds=manifolds_2)
    fs.check_conditions()
    sw_3.list_of_spacecraft = fs.list_of_sc

    sw_2 = swarm_1.particle_swarm(manifolds=manifolds, force_model=force_model_1)
    sw_2.integration_points = integration_points
    sw_2.sensitivity_swarm(center=[1.5e11, 0, 0, 0, 35000, 0])
    sw_2.integrate_swarm(rtol=1e-3, parproc=True, cores=5)
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
    force_model_1.central_attractor_gravity_law = kepler_dynamics.J_X_acceleration

    oe_vec = [15479300, 0.4, 60 * math.pi / 180, 100 * math.pi / 180, 26 * math.pi / 180, 206 * math.pi / 180]

    sc_1 = src.spacecraft.sc.Spacecraft(
        kepler_dynamics.oe_to_sv(oe_vec[0], oe_vec[1], oe_vec[2], oe_vec[3], oe_vec[4],
                                 oe_vec[5], 0, cent_mass), force_model=force_model_1)
    sc_2 = src.spacecraft.sc.Spacecraft(
        kepler_dynamics.oe_to_sv(oe_vec[0], oe_vec[1], oe_vec[2]+20*math.pi/180, oe_vec[3], oe_vec[4], oe_vec[5], 0, cent_mass),
        force_model=force_model_1)
    sc_3 = src.spacecraft.sc.Spacecraft(
        kepler_dynamics.oe_to_sv(oe_vec[0], oe_vec[1], oe_vec[2]+40*math.pi/180, oe_vec[3], oe_vec[4], oe_vec[5], 0, cent_mass),
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
    steps = 10000

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
    sw_1.integrate_swarm(rtol=1e-6, parproc=True, cores=5)
    sw_1.get_swarm_body_distances(body_list=["body_1"])
    list_of_spacecraft = sw_1.list_of_spacecraft
    list_of_spacecraft[0].plot_color = [1, 0.4, 1]

    fs = valid_set.feasibility_setup(list_of_sc=list_of_spacecraft, force_model=force_model_1, manifolds=[])
    fs.check_conditions()
    list_of_spacecraft = fs.list_of_sc

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=list_of_spacecraft,
                                            force_model=force_model_1, animated=False, axis_visibility=True, fps=None)

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
    sw_1.integrate_swarm(rtol=1e-10, parproc=True, cores=11)
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
    plots.moving_map_plot(match_tail_color=True, plot_central_attractor=False)


def CR3BP_ex_2():
    t_end = 20
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

    manifolds_1 = [[8.6121899389004331E-1, 8.6321899389004331E-1, 10],
                   [-6.1589472580081878E-28, -6.1589472580081878E-28, 1],
                   [-9.0996038678959414E-14, -9.0996038678959414E-14, 1],
                   [1.1168790093094281E-13, 1.1168790093094281E-13, 1],
                   [8.8512113504271353E-2, 8.8712113504271353E-2, 10],
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
    sw_1.integrate_swarm(rtol=1e-10, parproc=True, cores=11)
    sw_1.get_swarm_body_distances(body_list=["body_1", "body_2"])

    list_of_spacecraft = sw_1.list_of_spacecraft

    for sc in list_of_spacecraft:
        sc.plot_color = [1, 0.3, 1]

    fs = valid_set.feasibility_setup(manifolds=[], list_of_sc=list_of_spacecraft, force_model=force_model_1)
    fs.check_conditions()
    list_of_spacecraft = fs.list_of_sc

    input("Start plotting?")
    plots = plotting_functions.graph_output(list_of_spacecraft=[], list_of_resampled_spacecraft=[],
                                            list_of_special_spacecraft=list_of_spacecraft,
                                            force_model=force_model_1, animated=False, axis_visibility=False, fps=None)

    plots.state_space_slice(index=0, slices=[["x", "vy"]])
    plots.state_space_slice(index=999, slices=[["vx", "vy"], ["x", "y"]])
    plots.state_space_slice(index=999, slices=[["x", "y", "z"], ["vx", "vy", "vz"]])
    # plots.parameters_plot()
    # plots.C3_plot()
    plots.body_distances_plot(["body_1", "body_2"])
    plots.trajectory_xyz()
    # plots.moving_map_plot(match_tail_color=True, plot_central_attractor=False)
    plt.show()
    plt.waitforbuttonpress(10000000000)


# ex_7_SSO()
if __name__ == "__main__":
    # integrator_comparison()
    # ex_7()
    # Voyager_2()
    # ex_9()
    # set_intersection_counterexample()
    # dynamics_example_2d()
    # boundary_feasibility_base()
    # ex_3_integrator_comparison()
    # all_plots()
    # task_4_6()
    # orbiting_planet()
    CR3BP()
    # CR3BP_ex_2()
    # SSO()
