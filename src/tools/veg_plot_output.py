from sys import argv

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# hisfile_Spartina = nc.Dataset(r'C:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\sm_testcase6\output\VegModel_Spartina_his.nc', 'r')
# hisfile_Spartina.close
#
# hisfile_Salicornia = nc.Dataset(r'C:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\sm_testcase6\input\MinFiles\fm\cover\VegModel_Salicornia_his.nc', 'r')
# hisfile_Salicornia.close
#
# diaveg = hisfile_Salicornia.variables['diaveg'][:]
# diaveg = pd.DataFrame(data=diaveg)
# time = hisfile_Salicornia.variables['time'][:]
# time = pd.to_datetime(time)
# #
# veg_frac = hisfile_Salicornia.variables['veg_frac_j'][:]
# veg_frac = pd.DataFrame(data=veg_frac)
# veg_den = hisfile_Salicornia.variables['rnveg'][:]
# veg_den = pd.DataFrame(data = veg_den)
# veg_den = hisfile_Salicornia.variables['rnveg'][:]
# veg_den = pd.DataFrame(data = veg_den)
# veg_height = hisfile_Salicornia.variables['height'][:]
# veg_height = pd.DataFrame(data = veg_height)
#
# plt.plot(time, veg_height[121][:])
# plt.title("Vegetation height (Salicornia) and Fraction")
# plt.xlabel("time [years]")
# plt.ylabel("height [m]")
# plt.show()
#
# plt.plot(time, veg_frac[121][:])
# plt.show()
#
# # #
mapfile = nc.Dataset(
    r"C:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\sm_testcase6\input\MinFiles\fm\cover\VegModel_Salicornia_map.nc",
    "r",
)
mapfile.close

# veg_frac = mapfile.variables['veg_frac_j'][:]
# veg_frac = pd.DataFrame(data = veg_frac)
veg_den = mapfile.variables["rnveg"][-1, :].reshape(-1, 1)
veg_den = pd.DataFrame(data=veg_den)
veg_cover = mapfile.variables["cover"][-1, :].reshape(-1, 1)
veg_cover = pd.DataFrame(data=veg_cover)
time = mapfile.variables["time"][:]
veg_height = mapfile.variables["height"][-1, :].reshape(-1, 1)
veg_height = pd.DataFrame(data=veg_height)
x = mapfile.variables["nmesh2d_x"][:]
x = pd.DataFrame(data=x)
y = mapfile.variables["nmesh2d_y"][:]
y = pd.DataFrame(data=y)
bl = mapfile.variables["bl"][-1, :].reshape(-1, 1)
height_j = mapfile.variables["veg_height_j"][:]
height_j = pd.DataFrame(data=height_j[:, :, 0])
veg_frac_m = mapfile.variables["veg_frac_m"][:]
veg_frac_m = pd.DataFrame(data=veg_frac_m[:, :, -1])
veg_frac_j = mapfile.variables["veg_frac_j"][:]
veg_frac_j = pd.DataFrame(data=veg_frac_j[:, :, -1])
height_m = mapfile.variables["veg_height_m"][:]
height_m = pd.DataFrame(data=height_m[:, :, -1])
height_j = mapfile.variables["veg_height_j"][:]
height_j = pd.DataFrame(data=height_j[:, :, -1])
dia_m = mapfile.variables["veg_stemdia_m"][:]
dia_m = pd.DataFrame(data=dia_m[:, :, 0])

age_j = mapfile.variables["veg_age_j"][:]
age_j = pd.DataFrame(data=age_j[:, :, -1])

# # plt.scatter(x,y, c=veg_den)
# # cbar= plt.colorbar()
# # plt.show()
#
# # # plt.scatter(y,x, c=bl)
# # # # cbar= plt.colorbar()
# # # # plt.show()
# #
# # plt.scatter(y,x, c=veg_height)
# # cbar= plt.colorbar()
# # plt.show()
# #
plt.scatter(y, x, c=veg_cover)
cbar = plt.colorbar()
plt.title("Vegetation cover (Puccinellia)")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.show()
