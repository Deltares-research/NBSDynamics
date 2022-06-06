## libraries
# from sys import argv
#
# import matplotlib.cm as cm
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import netCDF4 as nc
import pandas as pd
import numpy as np
import os
# import cmocean
# import h5py
# import scipy.io as sio
#
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D


############################################# MATLAB
sim = 'testsg5'

for ets in range (1,24,1):
    blm = pd.read_csv(r'c:\Users\toledoal\uu_model\test30-05\{}\results_1\BL{}.csv'.format(sim, ets), skiprows=1, sep =',')
    blm = blm.dropna()
    blm1 = pd.read_csv(r'c:\Users\toledoal\uu_model\test30-05\{}\results_1\BL0.csv'.format(sim), skiprows=1, sep =',')
    blm1 = blm1.dropna()


    plt.figure()
    fig = plt.tricontourf(blm['x coordinate'],blm['y coordinate'],blm['bed level in water level points (m)']-blm1['bed level in water level points (m)'], cmap = 'terrain')
    plt.gca().set_aspect('equal')
    # cm = cmocean.cm.topo
    # bed.set_cmap(cm)
    bar =plt.colorbar(fig, spacing='uniform')
    bar.set_label('Bed level difference (m)')
    plt.title("Bed Level Difference (m)")
    plt.xlabel("Grid cell n-direction")
    plt.ylabel("Grid cell m-direction")

    plt.savefig(r'c:\Users\toledoal\uu_model\test30-05\{}\bedleveldif_ets{}'.format(sim, ets),dpi=300,bbox_inches='tight')
    plt.close()

    # blm['bed level in water level points (m)']*=(-1)
    plt.figure()
    fig = plt.tricontourf(blm['x coordinate'],blm['y coordinate'],blm['bed level in water level points (m)'],cmap='terrain')
    plt.gca().set_aspect('equal')
    # cm = cmocean.cm.topo
    # bed.set_cmap(cm)
    bar =plt.colorbar(fig, spacing='uniform', label = 'Bed level (m)' )
    plt.title("Bed Level (m)")
    plt.xlabel("Grid cell n-direction")
    plt.ylabel("Grid cell m-direction")

    vegpath = r'c:\Users\toledoal\uu_model\test30-05\{}\results_1\veg{}.trv'.format(sim,ets)
    if os.path.getsize(vegpath) > 0:
        veg = pd.read_csv(r'c:\Users\toledoal\uu_model\test30-05\{}\results_1\veg{}.trv'.format(sim,ets), header = None, sep ='\t')
        xp = ((veg[0]*34)-35)+2300
        yp = ((veg[1]*33))+1450

        # xp = ((veg[0]*35)-35)+2300
        # yp = ((veg[1]*35)-55)+1450

        # plt.figure()
        v =plt.scatter(yp,xp, c=veg[3], cmap = 'Greens')
        plt.gca().set_aspect('equal')
        # v.set_ylim(v.get_ylim()[::-1])
        # plt.gca().invert_yaxis()
        cbar = plt.colorbar(label = 'Fraction of Spartina (-)')


    plt.savefig(r'c:\Users\toledoal\uu_model\test30-05\{}\bedlevelandspart_ets{}'.format(sim, ets),dpi=300,bbox_inches='tight')
    plt.close()

##
sim = 'test30-05'
setup = 'morfac30visc5critero045sed300sedthr001eroadjcell1dif100'
time= '1701'
# bldifm= pd.read_csv(r'c:\Users\toledoal\uu_model\mud\d3d\results_2\CERO.csv', skiprows=1, sep =',')
# bldifm= pd.read_csv(r'c:\Users\toledoal\uu_model\{}\d3d\d3d4_{}\CERO.csv'.format(sim,sim), skiprows=1, sep =',')
bldifm= pd.read_csv(r'c:\Users\toledoal\uu_model\test30-05\CERO.csv', skiprows=1, sep =',')
bldifm = bldifm.dropna()
plt.figure()
ticks = np.arange(-0.008,0.028,0.004)
fig = plt.tricontourf(bldifm['x coordinate'],bldifm['y coordinate'],bldifm['cum. erosion/sedimentation (m)'], levels =ticks, cmap='terrain')
plt.gca().set_aspect('equal')
# cm = cmocean.cm.topo
# bed.set_cmap(cm)
bar =plt.colorbar(fig, spacing='uniform')
bar.set_label('Bed level (m)')
plt.title("Cumulative erosion and sedimentation (m)")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
# plt.savefig(r'c:\Users\toledoal\uu_model\{}\d3d\d3d4_{}\cumerosed_{}_{}.png'.format(sim,sim, setup, time),dpi=300,bbox_inches='tight')
plt.savefig(r'c:\Users\toledoal\uu_model\test30-05\cumerosed_{}_{}.png'.format(sim, time),dpi=300,bbox_inches='tight')

##
sim = 'testsg5'
ets=8
veg = pd.read_csv(r'c:\Users\toledoal\uu_model\test30-05\{}\results_1\veg{}.trv'.format(sim,ets), header = None, sep ='\t')
xp = (veg[0]*35)+1450
yp = (veg[1]*35)+2300

plt.figure()
v =plt.scatter(xp,yp y, c=veg[3], cmap = 'Greens')
plt.gca().set_aspect('equal')
# v.set_ylim(v.get_ylim()[::-1])
# plt.gca().invert_yaxis()
cbar = plt.colorbar()

data = f.get('data/variable1')
data = np.array(data) # For converting to a NumPy array

############################################# FLOW MAP
sim = 'bdv9'
# sed= 'sand'
mapflow = nc.Dataset(r"c:\Users\toledoal\NBSDynamicsD\test\test_data\{}\input\fm\output\{}_map.nc".format(sim,sim))
# mapflow = nc.Dataset(r"c:\Users\toledoal\NBSDynamicsD\test\test_data\tst4\30yearsrun-20yearswcf\output\FlowFMnowav_map.nc")
timeflow = mapflow.variables["time"][:]


wd = mapflow.variables['mesh2d_waterdepth'][-1,:]

x = mapflow.variables['mesh2d_face_x'][:]
y = mapflow.variables['mesh2d_face_y'][:]


blflow_first = mapflow.variables['mesh2d_mor_bl'][0,:]
# bl_flow_fev = mapflow.variables['mesh2d_mor_bl'][131,:]
#
# blflow_difd = bl_flow_fev -blflow_first

blflow_last = mapflow.variables['mesh2d_mor_bl'][-1,:]

blflow_diff = blflow_last - blflow_first

var = blflow_diff
plt.figure()
bed = plt.tricontourf(x,y,var)#, vmin=-8000, vmax=0, levels=blevels, cmap=cmap2, extend='both')
plt.gca().set_aspect('equal')
bed.set_cmap('terrain')
# cm = cmocean.cm.topo
# bed.set_cmap(cm)
bar =plt.colorbar(bed, spacing='uniform')
bar.set_label('Bed level difference (m)')
plt.title("Cumulative erosion and sedimentation (m)")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")

aaa = 1150
aa = round(((aaa*1200)/3600)/24)
# plt.savefig(r'c:\Users\toledoal\NBSDynamicsD\test\test_data\{}\input\fm\output\bldiff_ts{}'.format(sim,aa),dpi=300,bbox_inches='tight')
plt.savefig(r'c:\Users\toledoal\uu_model\test30-05\bldiff')
# plt.close()

##
for varfm in ['mesh2d_waterdepth','mesh2d_mor_bl', 'mesh2d_ucmag']:
    for i in range(340,len(timeflow),1):
        var = mapflow.variables[varfm][i, :]
        fig = plt.tricontourf(x, y, var)
        plt.gca().set_aspect('equal')
        plt.colorbar(fig, spacing='uniform')
        # plt.clim(0,2)
        figname = (r'c:\Users\toledoal\NBSDynamicsD\test\test_data\{}\input\fm\output\spa_{}_{}_ts{}'.format(sim,sed,varfm,str(i)))
        plt.savefig(figname,dpi=300,bbox_inches='tight')
        plt.close()

        # blflow = mapflow.variables['mesh2d_mor_bl'][i, :]
        # blf = plt.tricontourf(x, y, blflow, vmin=-1, vmax=1.5)
        # plt.gca().set_aspect('equal')
        # plt.colorbar(blf, spacing='uniform')

############################################# MAP VEG
# sim = r'tst4\30yearsrun-20yearswcf'
sim = r'FlowFM'
path =r"c:\Users\toledoal\NBSDynamicsD\test\test_data\{}\output".format(sim)
mapfile = nc.Dataset(path+r"\VegModel_Spartina_map.nc")#,
    # r"c:\Users\toledoal\NBSDynamicsD\test\test_data\sm_testcase6\input\MinFiles\fm\output_hydro_test1\FlowFM_map.nc",
#     "r",
# )
# mapfile.close

# Coordinates
x = mapfile.variables["nmesh2d_x"][:]
# x = pd.DataFrame(data=x)
y = mapfile.variables["nmesh2d_y"][:]
# y = pd.DataFrame(data=y)

wl_min = mapfile.variables['min_wl'][:]

wl_max = mapfile.variables['max_wl'][:]


veg_cover = mapfile.variables["cover"][-1, :]#.reshape(-1, 1)
veg_cover[veg_cover==0]=['nan']

veg_den = mapfile.variables["rnveg"][-1, :].reshape(-1, 1)
veg_den = pd.DataFrame(data=veg_den)


veg_cover = pd.DataFrame(data=veg_cover)





bl_last = mapfile.variables["bl"][-1, :]
bl_last = pd.DataFrame(data=bl_last)

bl_first = mapfile.variables["bl"][0, :]
bl_first = pd.DataFrame(data=bl_first)

bl_dif = bl_last-bl_first

bl = mapfile.variables["bl"][-1, :]
bl= pd.DataFrame(data=bl)






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

##PLOTS

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

plt.figure()
fig=plt.scatter(x, y, c=veg_cover)
cbar = plt.colorbar()
cbar.set_label('Vegetation cover (-)')
fig.set_cmap('plasma')
plt.title("Vegetation cover Spartina")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.gca().set_aspect('equal')


plt.scatter(x, y, c=bl)
cbar = plt.colorbar()
plt.title("Vegetation cover Spartina")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.gca().set_aspect('equal')



# fig, ax = plt.subplots(1,1, figsize=(10,6))
# ax.set_aspect('equal')
# xg,yg = np.meshgrid(x,y)
plt.figure()
ticks = np.arange(-1.6,1.3,0.1)
bed = plt.tricontourf(x,y,bl_last, vmin=-1.6,vmax =1.3, levels=ticks, cmap='terrain')#, vmin=-8000, vmax=0, levels=blevels, cmap=cmap2, extend='both')
plt.gca().set_aspect('equal')
bar = plt.colorbar(bed, label ='Bed Level (m)' )
bar.set_ticks(ticks)
# bar.set_label('Bed Level (m)')
# plt.clim(-1.6,1.2)
plt.title("Bed Level (m)")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")

fig=plt.scatter(x, y, c=veg_cover)
cbar = plt.colorbar()
cbar.set_label('Vegetation cover (-)')
fig.set_cmap('plasma')
plt.title("Vegetation cover Spartina")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.gca().set_aspect('equal')

figname = ( path+ r"\spa_bl_cover")
plt.savefig(figname, dpi=300, bbox_inches='tight')


for varfm in ['max_wl']:
    for i in range(2,512,10):
        var = mapfile.variables[varfm][i, :]
        fig = plt.tricontourf(x, y, var)
        plt.gca().set_aspect('equal')
        plt.colorbar(fig, spacing='uniform')
        # plt.clim(0,2)
        figname = (r'c:\Users\toledoal\NBSDynamicsD\test\test_data\tst4\30yearsrun-20yearswcf\spa_{}_ts{}'.format(varfm,str(i)))
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


time = mapfile.variables["time"][:]
veg_height = mapfile.variables["height"][:,1077]
veg_height = pd.DataFrame(data=veg_height)

height_m = mapfile.variables["veg_height_m"][:]
height_m = pd.DataFrame(data=height_m[1077, :, -1])
height_m = height_m.iloc[:,0]


timep= time.data()
timep= datetime.datetime.strptime(time, '%Y%M%D')



plt.figure()
plt.plot(veg_cover)
plt.title("Height and cover of Spartina in one cell")
plt.xticks(np.arange(0, 49, 4))
plt.xlabel("Time (ETS)")
plt.ylabel("Height (m)")
plt.plot(veg_cover)

## to write the dates
x.append(datetime.datetime.strptime(event['date'] + ' ' + event['time'] + ' ' + timeZone, '%Y-%m-%d %H:%M:%S %z'))
if event['value'] != missingValue:
    y.append(event['value'])
else:
    y.append(None)

dateFormatter = mdates.DateFormatter('%Y')
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(dateFormatter)
ax.xaxis.set_major_locator(plt.MaxNLocator(7))
ax.yaxis.set_major_locator(plt.MaxNLocator(8))
ax.grid(True)
##


plt.figure(1)
plt.plot(veg_stemdia_m)

plt.figure(2)
plt.plot(veg_frac_j)


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
