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
    max_u =  ma.MaskedArray.filled((mapfile.variables["max_u"][:, :]), 0.0)
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
    return bl, max_u, veg_den, veg_cover, time, veg_height, x, y, veg_frac_m, veg_frac_j, veg_frac_m, height_j, height_m, dia_m, dia_j, age_j, age_m

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
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    cbar= plt.colorbar(fig, label="Height [m]")
    plt.show()

def plot_cover_bath(x,y,bl,veg_cover):
    # PLOT vegetation cover over bed level
    fig = plt.tricontourf(x, y, bl[-1, :] , cmap = "terrain", levels=np.linspace(-1,4,80))
    cbar = plt.colorbar(fig, label="Bed level [m]")
    # plt.title("Vegetation cover (Spartina) and Bed Level [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")


    veg_cover[veg_cover==0] = np.nan
    FIG2  = plt.scatter(x, y, c=veg_cover[-1, :], cmap='Greens', edgecolors='k', lw = 0.2)
    cbar = plt.colorbar(FIG2,label="Fraction cover Elytrigia [-]")
    plt.title("Vegetation cover (Elytrigia) and Bed Level [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    plt.show()

    # mapfile = nc.Dataset(
    #     r"C:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Zuidgors_ref\output\VegModel_Salicornia_map.nc",
    #     "r",
    # )
    # veg_cover = ma.MaskedArray.filled((mapfile.variables["cover"][:, :]), 0.0)
    # veg_cover[veg_cover == 0] = np.nan
    # FIG2 = plt.scatter(x, y, c=veg_cover[-1, :], cmap='Blues', edgecolors='k', lw=0.2)
    # cbar = plt.colorbar(FIG2, label="Fraction cover Salicornia [-]")
    # plt.title("Vegetation cover (Salicornia) and Bed Level")
    # plt.xlabel("Grid cell x-direction")
    # plt.ylabel("Grid cell y-direction")
    # plt.show()

    #
    # mapfile = nc.Dataset(
    #     r"C:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Zuidgors_ref\output\VegModel_Spartina_map.nc",
    #     "r",
    # )
    # veg_cover = ma.MaskedArray.filled((mapfile.variables["cover"][:, :]), 0.0)
    # veg_cover[veg_cover == 0] = np.nan
    # FIG2 = plt.scatter(x, y, c=veg_cover[-1, :], cmap='Reds', edgecolors='k', lw=0.2)
    # cbar = plt.colorbar(FIG2, label="Fraction cover Spartina [-]")
    # plt.title("Vegetation cover (Spartina) and Bed Level")
    # plt.xlabel("Grid cell x-direction")
    # plt.ylabel("Grid cell y-direction")
    # plt.show()

def plot_bl_diff(x,y,bl, veg_cover):
# # BED LEVEL DIFFERENCE (sedimentation/ erosion)
    bl_diff = bl[-1, :] - bl[0, :]
    fig2 = plt.tricontourf(x, y, bl_diff , cmap = "terrain", levels=np.linspace(-1, 1, 80))
    cbar2 = plt.colorbar(fig2, label="Bed level difference [m]")
    plt.title("Bed Level Difference [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    plt.show()


    veg_cover[veg_cover == 0] = np.nan
    FIG2 = plt.scatter(x, y, c=veg_cover[-1, :], cmap='Greens', edgecolors='k', lw = 0.2)
    cbar = plt.colorbar(FIG2, label="Fraction cover [-]")
    plt.title("Vegetation cover (Spartina) and Bed Level Difference [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    plt.show()


def plot_max_vel(x, y, max_u):
    fig2 = plt.tricontourf(x, y, max_u[-1, :] , cmap = "viridis")
    cbar2 = plt.colorbar(fig2, label="Velocity [m/s]")
    plt.title("Max Flow Velocity")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    plt.show()

def make_gif():
    from celluloid import Camera  # getting the camera
    import matplotlib.pyplot as plt
    mpl.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'
    import numpy as np
    from IPython import display
    from IPython.display import HTML  # to show the animation in Jupyter
    import ffmpeg
    from matplotlib.animation import PillowWriter

    fig, ax = plt.subplots()  # creating my fig
    camera = Camera(fig)  # the camera gets the fig we'll plot
    for i in range(8):
        plt.tricontourf(x, y, max_u[i, :])  # WHAT NEEDS TO BE PLOTTED
        # cbar = plt.colorbar(fig, label="Velocity [m/s]")
        plt.title("Maximum Flow Velocity")
        plt.xlabel("Grid cell x-direction")
        plt.ylabel("Grid cell y-direction")
        camera.snap()  # the camera takes a snapshot of the plot
    animation = camera.animate(interval=1000, repeat=True,
                               repeat_delay=500)  # animation ready
    HTML(animation.to_html5_video())  # displaying the animation
    animation.save(
        r'C:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Zuidgors_new\figures\animation_vel.gif',
        writer='PillowWriter', fps=2)


mapfile = nc.Dataset(
    r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Zuidgors_ref\output\VegModel_Elytrigia_map.nc",
    "r",
)
bl, max_u, veg_den, veg_cover, time, veg_height, x, y, veg_frac_m, veg_frac_j, veg_frac_m, height_j, height_m, dia_m, dia_j, age_j, age_m = get_variables_mapfile(mapfile)
plot_cover_bath(x,y,bl,veg_cover)
# plot_veg_den(x, y, veg_den)
# plot_veg_height(x, y, veg_height)
# plot_bl_diff(x, y, bl, veg_cover)
# plot_max_vel(x, y, max_u)
# make_gif()
mapfile.close


# hisfile_Puccinellia = nc.Dataset(r'C:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\zuidgors_test\output_5yrs\VegModel_Salicornia_his.nc', 'r')
#
# diaveg, time, veg_frac, veg_den, veg_height, cover = get_variables_hisfile(hisfile_Puccinellia)
# plot_cover_time(time, cover, 641)
#
# hisfile_Puccinellia.close



