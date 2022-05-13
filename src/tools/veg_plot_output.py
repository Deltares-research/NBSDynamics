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


def get_variables_hisfile(hisfile):
    diaveg = hisfile.variables['diaveg'][:]
    diaveg = pd.DataFrame(data=diaveg)
    time = hisfile.variables['time'][:]
    time = pd.to_datetime(time)
    veg_frac = hisfile.variables['veg_frac_j'][:]
    veg_frac = pd.DataFrame(data=veg_frac)
    veg_den = hisfile.variables['rnveg'][:]
    veg_den = pd.DataFrame(data=veg_den)
    veg_den = hisfile.variables['rnveg'][:]
    veg_den = pd.DataFrame(data=veg_den)
    veg_height = hisfile.variables['height'][:]
    veg_height = pd.DataFrame(data=veg_height)
    cover = hisfile.variables['cover'][:]
    cover = pd.DataFrame(data=cover)
    return diaveg, time, veg_frac, veg_den, veg_height, cover

def plot_VegHeight_time(time, veg_height, cell):
    plt.plot(time, veg_height.iloc[cell, :])
    plt.title("Vegetation height and Cover")
    plt.xlabel("time [years]")
    plt.ylabel("height [m]")
    plt.show()

def plot_cover_time(time, cover, cell):
    plt.plot(time, cover.iloc[cell, :])
    plt.show()


def get_variables_mapfile(mapfile):
    bl = ma.MaskedArray.filled((mapfile.variables["bl"][:, :]), 0.0) # Bed level
    veg_den = ma.MaskedArray.filled((mapfile.variables["rnveg"][:, :]), 0.0) # average stem diameter each cell
    veg_cover = ma.MaskedArray.filled((mapfile.variables["cover"][:, :]), 0.0) # Vegetation Cover Fraction
    time = ma.MaskedArray.filled((mapfile.variables["time"][:]), 0.0) # Time
    veg_height = ma.MaskedArray.filled((mapfile.variables["height"][:, :]), 0.0) # average vegetation height
    x = ma.MaskedArray.filled((mapfile.variables["nmesh2d_x"][:]), 0.0) # x-coordinated
    y = ma.MaskedArray.filled((mapfile.variables["nmesh2d_y"][:]), 0.0) # y-coordinates
    veg_frac_m = ma.MaskedArray.filled((mapfile.variables["veg_frac_m"][:, :, :]), 0.0)
    veg_frac_j = ma.MaskedArray.filled((mapfile.variables["veg_frac_j"][:, :, :]), 0.0)
    height_m = ma.MaskedArray.filled((mapfile.variables["veg_height_m"][:, :, :]), 0.0)
    height_j = ma.MaskedArray.filled((mapfile.variables["veg_height_j"][:, :, :]), 0.0)
    dia_j = ma.MaskedArray.filled((mapfile.variables["veg_stemdia_j"][:, :, :]), 0.0)
    dia_m = ma.MaskedArray.filled((mapfile.variables["veg_stemdia_m"][:, :, :]), 0.0)
    age_j = ma.MaskedArray.filled((mapfile.variables["veg_age_j"][:, :, :]), 0.0)
    age_m = ma.MaskedArray.filled((mapfile.variables["veg_age_m"][:, :, :]), 0.0)
    return bl, veg_den, veg_cover, time, veg_height, x, y, veg_frac_m, veg_frac_j, veg_frac_m, height_j, height_m, dia_m, dia_j, age_j, age_m

def plot_veg_den(x,y,veg_den):
    fig = plt.tricontourf(x, y, veg_den[-1, :])
    cbar = plt.colorbar(fig, label="Density [stem/m2]")
    plt.title("Vegetation Density")
    plt.xlabel("Grid cell n-direction")
    plt.ylabel("Grid cell m-direction")
    plt.show()

def plot_veg_height(x,y,veg_height):
    fig = plt.scatter(x,y, c=veg_height[-1, :])
    plt.title("Average Vegetation Height")
    plt.xlabel("Grid cell n-direction")
    plt.ylabel("Grid cell m-direction")
    cbar= plt.colorbar(fig, label="Height [m]")
    plt.show()

def plot_cover_bath(x,y,bl,veg_cover):
    # PLOT vegetation cover over bed level
    fig = plt.tricontourf(x, y, bl[-1, :] , cmap = "terrain", levels=np.linspace(0,4,60))
    cbar = plt.colorbar(fig, label="Bed level [m]")
    # plt.title("Vegetation cover (Spartina) and Bed Level [m]")
    plt.xlabel("Grid cell n-direction")
    plt.ylabel("Grid cell m-direction")


    veg_cover[veg_cover==0] = np.nan
    FIG2  = plt.scatter(x, y, c=veg_cover[-1, :], cmap='YlGn')
    cbar = plt.colorbar(FIG2,label="Fraction cover [-]")
    plt.title("Vegetation cover (Spartina) and Bed Level [m]")
    plt.xlabel("Grid cell n-direction")
    plt.ylabel("Grid cell m-direction")
    plt.show()

def plot_bl_diff(x,y,bl, veg_cover):
# # BED LEVEL DIFFERENCE (sedimentation/ erosion)
    bl_diff = bl[-1, :] - bl[0, :]
    fig2 = plt.tricontourf(x, y, bl_diff , cmap = "terrain", levels=np.linspace(-1, 1, 80))
    cbar2 = plt.colorbar(fig2, label="Bed level difference [m]")
    plt.title("Bed Level Difference [m]")
    plt.xlabel("Grid cell n-direction")
    plt.ylabel("Grid cell m-direction")
    plt.show()


    veg_cover[veg_cover == 0] = np.nan
    FIG2 = plt.scatter(x, y, c=veg_cover[-1, :], cmap='YlGn')
    cbar = plt.colorbar(FIG2, label="Fraction cover [-]")
    plt.title("Vegetation cover (Spartina) and Bed Level Difference [m]")
    plt.xlabel("Grid cell n-direction")
    plt.ylabel("Grid cell m-direction")
    plt.show()

mapfile = nc.Dataset(
    r"C:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\Zuidgors_new\output\VegModel_Spartina_map.nc",
    "r",
)
bl, veg_den, veg_cover, time, veg_height, x, y, veg_frac_m, veg_frac_j, veg_frac_m, height_j, height_m, dia_m, dia_j, age_j, age_m = get_variables_mapfile(mapfile)
plot_cover_bath(x,y,bl,veg_cover)
# plot_veg_den(x, y, veg_den)
# plot_veg_height(x, y, veg_height)
# plot_bl_diff(x, y, bl, veg_cover)
mapfile.close


# hisfile_Puccinellia = nc.Dataset(r'C:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\zuidgors_test\output_5yrs\VegModel_Salicornia_his.nc', 'r')
#
# diaveg, time, veg_frac, veg_den, veg_height, cover = get_variables_hisfile(hisfile_Puccinellia)
# plot_cover_time(time, cover, 641)
#
# hisfile_Puccinellia.close
