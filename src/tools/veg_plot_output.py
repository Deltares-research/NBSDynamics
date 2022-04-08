
import netCDF4 as nc
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sys import argv
import matplotlib.pyplot as plt

import matplotlib.cm as cm
hisfile = nc.Dataset(r'C:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\sm_testcase\output\VegModel_his.nc', 'r')
hisfile.close

diaveg = hisfile.variables['diaveg'][:]
diaveg = pd.DataFrame(data = diaveg)
time = hisfile.variables['time'][:]
time = pd.to_datetime(time)
#
veg_frac = hisfile.variables['veg_frac_j'][:]
veg_frac = pd.DataFrame(data = veg_frac)
veg_den = hisfile.variables['rnveg'][:]
veg_den = pd.DataFrame(data = veg_den)
veg_den = hisfile.variables['rnveg'][:]
veg_den = pd.DataFrame(data = veg_den)
veg_height = hisfile.variables['height'][:]
veg_height = pd.DataFrame(data = veg_height)

plt.plot(time, veg_height[135][:])
plt.show()
#
plt.plot(time, veg_den[130][:])
plt.show()
#

# mapfile = nc.Dataset(r'C:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\sm_testcase\output\VegModel_map.nc', 'r')
# mapfile.close
#
# # veg_frac = mapfile.variables['veg_frac_j'][:]
# # veg_frac = pd.DataFrame(data = veg_frac)
# veg_den = mapfile.variables['rnveg'][-1, :].reshape(-1,1)
# # veg_den = pd.DataFrame(data = veg_den)
# veg_cover = mapfile.variables['cover'][:]
# veg_cover = pd.DataFrame(data = veg_cover)
# time = mapfile.variables['time'][:]
# veg_height = mapfile.variables['height'][9 , :].reshape(-1,1)
# # veg_height = pd.DataFrame(data = veg_height)
# x = mapfile.variables['nmesh2d_x'][:]
# x = pd.DataFrame(data = x)
# y = mapfile.variables['nmesh2d_y'][:]
# y = pd.DataFrame(data = y)
# bl = mapfile.variables['bl'][-1, :].reshape(-1, 1)


# plt.scatter(x,y, c=veg_den)
# cbar= plt.colorbar()
# plt.show()

# plt.scatter(y,x, c=bl)
# # cbar= plt.colorbar()
# # plt.show()
#
# plt.scatter(y,x, c=veg_height)
# cbar= plt.colorbar()
# plt.show()
