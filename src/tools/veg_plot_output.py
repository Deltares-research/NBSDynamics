from sys import argv

import matplotlib.cm as cm
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import netCDF4 as nc
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

####################### HIS

hisfile_Spartina2 = nc.Dataset(r'c:\Users\toledoal\NBSDynamicsD\test\test_data\sm_testcase_mud\outputtestsarah2\VegModel_Spartina_his.nc', 'r')
# hisfile_Salicornia.close

diaveg = hisfile_Spartina.variables['diaveg'][:]
diaveg = pd.DataFrame(data = diaveg)
time = hisfile_Spartina.variables['time'][:]
time = pd.to_datetime(time)
#
veg_frac_j = hisfile_Spartina2.variables['veg_frac_j'][:]
veg_frac_j = pd.DataFrame(data = veg_frac)

veg_den = hisfile_Spartina.variables['rnveg'][:]
veg_den = pd.DataFrame(data = veg_den)
veg_den = hisfile_Spartina.variables['rnveg'][:]
veg_den = pd.DataFrame(data = veg_den)
veg_height = hisfile_Spartina.variables['height'][:]
veg_height = pd.DataFrame(data = veg_height)

maxtau = hisfile_Spartina.variables['max_tau'][:]
# plt.plot(time, veg_height[93][:])
# plt.title("Vegetation height (Spartina)")
# plt.xlabel("time [years]")
# plt.ylabel("height [m]")
# plt.show()

# plt.plot(time, veg_frac[125][:])
# plt.show()

############################# FLOW MAP

mapflow = nc.Dataset(r"c:\Users\toledoal\NBSDynamicsD\test\test_data\fm_10x30_sand2\input\MinFiles\fm\output\test_case_sand3_map.nc")
timeflow = mapflow.variables["time"][:]


wd = mapflow.variables['mesh2d_waterdepth'][:,152]


blflow = mapflow.variables['mesh2d_mor_bl'][0,:]



############################## MAP

mapfile = nc.Dataset(
    r"c:\Users\toledoal\NBSDynamicsD\test\test_data\fm_10x30_sand2\output\VegModel_Spartina_map.nc",
    # r"c:\Users\toledoal\NBSDynamicsD\test\test_data\sm_testcase6\input\MinFiles\fm\output_hydro_test1\FlowFM_map.nc",
    "r",
)
# mapfile.close


veg_den = mapfile.variables["rnveg"][-1, :].reshape(-1, 1)
veg_den = pd.DataFrame(data=veg_den)

veg_cover = mapfile.variables["cover"][-1, :]#.reshape(-1, 1)
veg_cover = pd.DataFrame(data=veg_cover)

time = mapfile.variables["time"][:]
veg_height = mapfile.variables["height"][9, :].reshape(-1, 1)
veg_height = pd.DataFrame(data=veg_height)

x = mapfile.variables["nmesh2d_x"][:]
# x = pd.DataFrame(data=x)
y = mapfile.variables["nmesh2d_y"][:]
# y = pd.DataFrame(data=y)

bl_last = mapfile.variables["bl"][-1, :]
bl_last = pd.DataFrame(data=bl_last)

bl_first = mapfile.variables["bl"][0, :]
bl_first = pd.DataFrame(data=bl_first)

bl_dif = bl_last-bl_first

bl = mapfile.variables["bl"][-1, :]
bl= pd.DataFrame(data=bl)

wl_min = mapfile.variables['min_wl'][:]

wl_max = mapfile.variable['max_wl'][]

height_m = mapfile.variables["veg_height_m"][:]
height_m = pd.DataFrame(data=height_m[:, :, -1])
height_m = height_m.iloc[:,0]


age_j = mapfile.variables["age_j"][:]
age_j = pd.DataFrame(data=age_j)

veg_age_m = mapfile.variables["veg_age_m"][:]
veg_age_m = pd.DataFrame(data=veg_age_m[:,:,--2])
veg_age_m = pd.DataFrame(data=veg_age_m[241,0,:])

veg_frac_j = mapfile.variables["veg_frac_j"][:,:,-1]
veg_frac_j = pd.DataFrame(data=veg_frac_j[8,3, -1])
veg_frac_j = pd.DataFrame(data=veg_frac_j[:,:, -1])

veg_frac_m = mapfile.variables["veg_frac_m"][:,:,-1]
veg_frac_m = pd.DataFrame(data=veg_frac_m[12, 15, :])
veg_frac_m = pd.DataFrame(data=veg_frac_m[:, :, -1])

veg_stemdia_m = mapfile.variables["veg_stemdia_m"][:]
veg_stemdia_m = pd.DataFrame(data=veg_stemdia_m[241, 0, :])

veg_age_j = mapfile.variables["veg_age_j"][:]
veg_age_j = pd.DataFrame(data=veg_age_j[:,:,-2])
veg_age_j = pd.DataFrame(data=veg_age_j[8,3,:])

height_m = mapfile.variables["veg_height_m"][:]
height_m = pd.DataFrame(data=height_m[42,17, :])
# plt.scatter(x,y, c=veg_den)
# cbar= plt.colorbar()
# plt.show()

# # plt.scatter(y,x, c=bl)
# # # cbar= plt.colorbar()
# # # plt.show()

######################## PLOTS

plt.scatter(x,y, c=height_m)
cbar= plt.colorbar()
plt.show()


plt.scatter(x,y, c=veg_frac_j, cmap= 'Greens')
plt.figure()
plt.scatter(x,y, c=veg_frac_m, cmap= 'Oranges')
plt.gca().set_aspect('equal')
cbar= plt.colorbar()
plt.show()


plt.figure()
plt.scatter(x, y, c=bl_last)
plt.gca().set_aspect('equal')
cbar = plt.colorbar()

plt.figure()
plt.scatter(x, y, c=bl_first)
plt.gca().set_aspect('equal')
cbar = plt.colorbar()

plt.scatter(x, y, c=veg_cover)
cbar = plt.colorbar()
plt.title("Vegetation cover Spartina")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.gca().set_aspect('equal')
plt.show()


plt.scatter(x, y, c=bl)
cbar = plt.colorbar()
plt.title("Vegetation cover Spartina")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.gca().set_aspect('equal')
plt.show()
plt.gca().set_aspect('equal')


# fig, ax = plt.subplots(1,1, figsize=(10,6))
# ax.set_aspect('equal')
# xg,yg = np.meshgrid(x,y)
bed = plt.tricontourf(x,y,bl_dif)#, vmin=-8000, vmax=0, levels=blevels, cmap=cmap2, extend='both')
ax.set_aspect('equal')
plt.colorbar(bed, spacing='uniform')
plt.title("Bed Level (m)")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")


for varfm in ['mesh2d_waterdepth','mesh2d_mor_bl']:
    for i in range(7000,7700,100):
        var = mapflow.variables[varfm][i, :]
        fig = plt.tricontourf(x, y, var)
        plt.gca().set_aspect('equal')
        plt.colorbar(fig, spacing='uniform')
        # plt.clim(0,2)
        figname = (r'c:\Users\toledoal\NBSDynamicsD\test\test_data\fm_10x30_sand2\output\sali_{}_ts{}'.format(varfm,str(i)))
        plt.savefig(figname,dpi=300,bbox_inches='tight')
        plt.close()

        blflow = mapflow.variables['mesh2d_mor_bl'][i, :]
        blf = plt.tricontourf(x, y, blflow, vmin=-1, vmax=1.5)
        plt.gca().set_aspect('equal')
        plt.colorbar(blf, spacing='uniform')




plt.subplots(1,2)
wdep = plt.tricontourf(x,y,wd)
plt.colorbar(wdep, spacing='uniform')
# veg_covernan= [veg_cover][[veg_cover>0] == np.nan]

plt.scatter(x, y, c=veg_cover,cmap='Greens')

cover = plt.tricontourf(x,y,veg_cover, cmap= 'Greens')#, vmin=-8000, vmax=0, levels=blevels, cmap=cmap2, extend='both')
plt.imshow(cover)#, cmap='Greens')
plt.colorbar(cover, spacing='uniform')



fig, ax = plt.subplots(1,1, figsize=(10,6))
wdep = plt.tricontourf(x,y,wd)
plt.colorbar(wdep, spacing='uniform')


plt.plot(veg_frac_m)
plt.show()

plt.figure(1)
plt.plot(height_m)

plt.figure(1)
plt.plot(veg_stemdia_m)

plt.figure(2)
plt.plot(veg_frac_j)