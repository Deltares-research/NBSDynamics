
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import numpy.ma as ma


mapfile = nc.Dataset(
    r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Zuidgors_bigger_mangroves\output\MangroveModel_map.nc",
    "r",
)
bl = ma.MaskedArray.filled((mapfile.variables["bl"][:, :]), 0.0) # Bed level
biomass = ma.MaskedArray.filled((mapfile.variables['tot_biomass'][:]), 0.0) # total mangrove biomass in cell
x = ma.MaskedArray.filled((mapfile.variables["nmesh2d_x"][:]), 0.0)  # x-coordinated
y = ma.MaskedArray.filled((mapfile.variables["nmesh2d_y"][:]), 0.0)  # y-coordinates
height = ma.MaskedArray.filled((mapfile.variables["height"][:, :, -1]), 0.0)
stem_num = ma.MaskedArray.filled((mapfile.variables["stem_num"][:, :, -1]), 0.0)
stem_dia = ma.MaskedArray.filled((mapfile.variables["stem_dia"][:, :, -1]), 0.0)
av_height = np.sum((stem_num * height), axis=1)/ np.sum(stem_num, axis=1) #without roots
cum_stem_dia = np.sum((stem_num * stem_dia), axis=1)
# sum_stem_num = np.sum(stem_num, axis= 1)

# PLOT vegetation cover over bed level
fig = plt.tricontourf(x, y, bl[-1, :], cmap="terrain", levels=np.linspace(-1, 4, 80))
cbar = plt.colorbar(fig, label="Bed level [m]")
plt.title("Mangrove biomass and Bed Level [m]")
plt.xlabel("Grid cell x-direction")
plt.ylabel("Grid cell y-direction")

biomass[biomass == 0] = np.nan
FIG2 = plt.scatter(x, y, c=biomass[-1, :], cmap='Greens', edgecolors='k', lw=0.2)
cbar = plt.colorbar(FIG2, label="Total biomass")
plt.xlabel("Grid cell x-direction")
plt.ylabel("Grid cell y-direction")
plt.show()

