
import netCDF4 as nc
import pandas as pd
from matplotlib import cm
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

hisfile = nc.Dataset(r'c:\Users\toledoal\sm_testcase2_Pdrive\MinFiles\fm\output\VegModel_his.nc', 'r')
# hisfile.close

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

plt.plot(time, veg_height[129][:])
plt.show()
#
plt.plot(time, veg_frac[129][:])
plt.show()


%% MAP

mapfile = nc.Dataset(r'c:\Users\toledoal\sm_testcase2_Pdrive\MinFiles\fm\output\VegModel_map.nc', 'r')
# mapfile.close

# veg_frac = mapfile.variables['veg_frac_j'][:]
# veg_frac = pd.DataFrame(data = veg_frac)
veg_den = mapfile.variables['rnveg'][-1, :].reshape(-1,1)
veg_den1 = mapfile.variables['rnveg'][-1, :]
veg_den2 = mapfile.variables['rnveg']
# veg_den = pd.DataFrame(data = veg_den)
veg_cover = mapfile.variables['cover'][:]
veg_cover = pd.DataFrame(data = veg_cover)
time = mapfile.variables['time'][:]
veg_height = mapfile.variables['height'][9 , :].reshape(-1,1)
# veg_height = pd.DataFrame(data = veg_height)
x = mapfile.variables['nmesh2d_x'][:]
x = pd.DataFrame(data = x)
y = mapfile.variables['nmesh2d_y'][:]
y = pd.DataFrame(data = y)
bl = mapfile.variables['bl'][-1, :].reshape(-1, 1)


plt.scatter(x,y, c=veg_den)
cbar= plt.colorbar()
plt.show()

plt.scatter(y,x, c=bl)
# cbar= plt.colorbar()
# plt.show()

plt.scatter(y,x, c=veg_height)
cbar= plt.colorbar()
plt.show()
