from sys import argv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
import matplotlib.animation as ani

mapfile = nc.Dataset(
    r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\sm_testcase_mud\output\MangroveModel_map.nc",
    "r",
)
bl = ma.MaskedArray.filled((mapfile.variables["bl"][:, :]), 0.0) # Bed level
biomass = mapfile.variables['tot_biomass'][:]
x = ma.MaskedArray.filled((mapfile.variables["nmesh2d_x"][:]), 0.0)  # x-coordinated
y = ma.MaskedArray.filled((mapfile.variables["nmesh2d_y"][:]), 0.0)  # y-coordinates

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

