import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy.ma as ma
import datetime
import netCDF4 as nc
import pandas as pd
import numpy as np
import os


## MAP VEG

sim = 'FlowFM2'
sp = 'Puccinellia'
ets = -1
path =r"c:\Users\toledoal\NBSDynamicsD\test\test_data\{}\output".format(sim)
mapfile = nc.Dataset(path+r"\VegModel_{}_map.nc".format(sp))

# Coordinates
x = mapfile.variables["nmesh2d_x"][:]
# x = pd.DataFrame(data=x)
y = mapfile.variables["nmesh2d_y"][:]
# y = pd.DataFrame(data=y)

# Variables
wl_min = mapfile.variables['min_wl'][:]

wl_max = mapfile.variables['max_wl'][:]

bl_last = mapfile.variables["bl"][ets, :]

bl_first = mapfile.variables["bl"][0, :]

bl_dif = bl_last-bl_first

veg_cover = mapfile.variables["cover"][ets, :]
veg_cover[veg_cover==0]=['nan']
aaa = pd.DataFrame(data = veg_cover)

veg_den = mapfile.variables["rnveg"][ets, :]

stemdia = mapfile.variables["diaveg"][ets, :]

veg_height = mapfile.variables["height"][ets, :]

## MAP PLOTS


plt.figure()
ticks = np.arange(-1.0,1.3,0.2)
bed = plt.tricontourf(x,y,bl_last, levels=ticks, cmap='terrain')#, vmin=-8000, vmax=0, levels=blevels, cmap=cmap2, extend='both')
# plt.gca().set_aspect('equal')
bar = plt.colorbar(bed, label ='Bed Level (m)' )
bar.set_ticks(ticks)
# bar.set_label('Bed Level (m)')
# plt.clim(-1.6,1.2)
plt.title("Bed Level (m)")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")

var = veg_cover
fig = plt.scatter(x, y, var)
cbar = plt.colorbar()
cbar.set_label('Vegetation cover (-)')
fig.set_cmap('Greens')
plt.title("{} Vegetation Cover".format(sp))
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
# plt.gca().set_aspect('equal')

figname = ( path+ r"\{}_bl_cover".format(sp))
plt.savefig(figname, dpi=300, bbox_inches='tight')
# plt.close()

## Time series plots of height and vegetation cover

time = mapfile.variables["time"][:]
time = time.tolist()

veg_cover = mapfile.variables["cover"][ets, :]
veg_cover = pd.DataFrame(data=veg_cover)

cell = 222
veg_cover = mapfile.variables["cover"][:, cell]
veg_height = mapfile.variables["height"][:, cell]

plt.figure()
plt.plot(veg_cover, label = 'Cover (-)')
plt.plot(veg_height, label = 'Height (m)')
plt.xticks(np.arange(0,24,2),time[::2], rotation =45)
plt.legend(loc='best', shadow=True)
plt.title("Height and cover of {} in one cell".format(sp))
plt.xlabel("Time (Date)")
plt.ylabel("Height (m) and cover (-)")

figname = ( path+ r"\{}_height_cover_{}".format(sp,cell))
plt.savefig(figname, dpi=300, bbox_inches='tight')



