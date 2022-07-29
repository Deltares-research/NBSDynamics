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
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.interpolate import griddata

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

# X,Y = np.meshgrid(x, y)
# Z_Height = griddata((x, y), av_height, (X,Y), method='nearest')
# BL = griddata((x, y), bl[-1, :], (X,Y), method='nearest')
# CUM_stem = griddata((x, y), cum_stem_dia, (X,Y), method='nearest')
# x1, y1, z1, Bl, CUM_STEM = X.ravel(), Y.ravel(), Z_Height.ravel(), BL.ravel(), CUM_stem.ravel()
#
# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')
# # x1 = x[np.isnan(biomass[-1, :]) == False]
# # y1 = y[np.isnan(biomass[-1, :]) == False]
# ax.plot_trisurf(x, y, bl[-1, :], cmap = 'terrain')
#
# bottom = np.zeros(z1.shape)
# width = np.ones(z1.shape)
# depth = np.ones(z1.shape)
# ax.bar3d(x1[z1>0], y1[z1>0], Bl[z1>0], (CUM_STEM[z1>0]/100), (CUM_STEM[z1>0]/100), (z1[z1>0]+ Bl[z1>0]), color = 'g')
#
#
# ax.set_xlabel('x axis')
# ax.set_ylabel('y axis')
# ax.set_zlabel('height [m]')
# ax.elev = 23
# ax.azim = -11
#
# plt.show()
# ax.stem(x1, y1, bl[-1, :][np.isnan(biomass[-1, :]) == False])





# Interpolate (x,y,z) points [mat] over a normal (x,y) grid [X,Y]
#   Depending on your "error", you may be able to use other methods

#
# plt.pcolormesh(X,Y,Z)
# plt.show()



# ax = fig.add_subplot(111, projection='3d')

# for xi, yi, zi in zip(x, y, z):
#     line = art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), marker='o', markevery=(1,1))
#     # if zi > 0:
#     #     a = 1
#     ax.add_line(line)
#
#
# plt.show()

# ax = fig.add_subplot(111, projection='3d')
