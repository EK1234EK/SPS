import matplotlib.pyplot as plt
import pandas
import pandas as pd
import numpy as np
import math
import matplotlib as mpl

def get_matrix(DF: pandas.DataFrame):
    sigma = DF["sigma"].tolist()
    TAEPO = DF["TAEPO"].tolist()
    alt = DF["alt"].tolist()

    pattern = dict()
    for i in range(len(alt)):
        pattern[(round(sigma[i], 3), round(TAEPO[i], 3))] = alt[i]

    nval_sigma = len(set(sigma))
    nval_TAEPO = len(set(TAEPO))

    sigma_axis = np.linspace(min(sigma), max(sigma), nval_sigma)
    TAEPO_axis = np.linspace(min(TAEPO), max(TAEPO), nval_TAEPO)

    sigma_grid, TAEPO_grid = np.meshgrid(sigma_axis, TAEPO_axis)
    alt_grid = np.zeros((nval_sigma, nval_TAEPO))
    for i in range(nval_sigma):
        for j in range(nval_TAEPO):
            sigma_idx = round(sigma_axis[i], 3)
            TAEPO_idx = round(TAEPO_axis[j], 3)
            alt_grid[i][j] = pattern[(sigma_idx, TAEPO_idx)]
    return sigma_grid, TAEPO_grid, alt_grid

armando_data = pd.read_excel("../Altitude_data_Armando.xlsx")

print(min(armando_data["alt"].tolist()))
exit

calc_data = pd.read_excel("../Altitude_cutoff_data.xlsx")

sigma_grid_armando, TAEPO_grid_armando, alt_grid_armando = get_matrix(armando_data)

sigma_grid_calc, TAEPO_grid_calc, alt_grid_calc = get_matrix(calc_data)

fig = plt.figure()
ax_1 = fig.add_subplot(121, projection="3d")
ax_2 = fig.add_subplot(122, projection="3d")
im_1 = ax_1.plot_surface(sigma_grid_armando, TAEPO_grid_armando, alt_grid_armando, cmap=mpl.colormaps["hsv"])
im_2 = ax_2.plot_surface(sigma_grid_calc, TAEPO_grid_calc, alt_grid_calc, cmap=mpl.colormaps["hsv"])

for ax in [ax_1, ax_2]:
    ax.set_xlabel("Sail loading [kg / m^2]")
    ax.set_ylabel("Init anomaly [°]")
    ax.set_zlabel("Minimum altitude [km]")
    ax.view_init(elev=25, azim=135)

fig.colorbar(im_2, orientation='vertical')

ax_1.set_title("Armando")
ax_2.set_title("Calculated")
plt.show()