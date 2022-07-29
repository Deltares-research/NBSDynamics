from sys import argv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib as mpl

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
    fig = plt.figure(figsize=(12, 8))
    plt.plot(time, veg_height.iloc[cell, :])
    plt.title("Vegetation height and Cover")
    plt.xlabel("time [years]")
    plt.ylabel("height [m]")
    plt.show()

def plot_cover_time(time, cover, cell):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(time, cover.iloc[cell, :])
    plt.show()


def get_variables_mapfile(mapfile):
    bl = ma.MaskedArray.filled((mapfile.variables["bl"][:, :]), 0.0) # Bed level
    max_u =  ma.MaskedArray.filled((mapfile.variables["max_u"][:, :]), 0.0)
    veg_den = ma.MaskedArray.filled((mapfile.variables["rnveg"][:, :]), 0.0) # average stem diameter each cell
    veg_dia = ma.MaskedArray.filled((mapfile.variables["diaveg"][:, :]), 0.0)  # average stem diameter each cell
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
    return bl, max_u, veg_den, veg_dia, veg_cover, time, veg_height, x, y, veg_frac_m, veg_frac_j, veg_frac_m, height_j, height_m, dia_m, dia_j, age_j, age_m

def plot_veg_den(x,y,veg_den):
    fig = plt.figure(figsize=(12, 8))
    fig = plt.tricontourf(x, y, veg_den[-1, :])
    cbar = plt.colorbar(fig, label="Density [stem/m2]")
    plt.title("Vegetation Density")
    plt.xlabel("Grid cell n-direction")
    plt.ylabel("Grid cell m-direction")
    plt.show()

def plot_veg_height(x,y,veg_height):
    fig = plt.figure(figsize=(12, 8))
    fig = plt.scatter(x,y, c=veg_height[-1, :])
    plt.title("Average Vegetation Height")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    cbar= plt.colorbar(fig, label="Height [m]")
    plt.show()

def plot_cover_bath(x,y,bl,veg_cover):
    # PLOT vegetation cover over bed level
    fig = plt.figure(figsize=(12, 8))
    fig = plt.tricontourf(x, y, bl[-1, :] , cmap = "terrain", levels=np.linspace(0,4,80))
    cbar = plt.colorbar(fig, label="Bed level [m]")
    # plt.title("Vegetation cover (Spartina) and Bed Level [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")

    veg_cove = veg_cover.copy()
    veg_cove[veg_cover==0] = np.nan
    FIG2  = plt.scatter(x, y, c=veg_cove[-1, :], cmap='Greens', edgecolors='k', lw = 0.2)
    cbar = plt.colorbar(FIG2,label="Fraction cover Spartina [-]")
    plt.title("Vegetation cover (Spartina) and Bed Level [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    plt.show()
    #
    # mapfile2 = nc.Dataset(
    #     r"c:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\Zuidgors_bigger\output\VegModel_Puccinellia_map.nc",
    #     "r",
    # )
    # veg_cover2 = ma.MaskedArray.filled((mapfile2.variables["cover"][:, :]), 0.0)
    # veg_cove2 = veg_cover2.copy()
    # veg_cove2[veg_cover2 == 0] = np.nan
    # FIG2 = plt.scatter(x, y, c=veg_cove2[-1, :], cmap='Blues', edgecolors='k', lw=0.2)
    # cbar = plt.colorbar(FIG2, label="Fraction cover Puccinellia [-]")
    # plt.title("Vegetation cover (Puccinellia and Spartina) and Bed Level")
    # plt.xlabel("Grid cell x-direction")
    # plt.ylabel("Grid cell y-direction")
    # plt.show()
    #
    # #
    # mapfile3 = nc.Dataset(
    #     r"c:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\Zuidgors_bigger\output\VegModel_Elytrigia_map.nc",
    #     "r",
    # )
    # veg_cover3 = ma.MaskedArray.filled((mapfile3.variables["cover"][:, :]), 0.0)
    # veg_cove3 = veg_cover3.copy()
    # veg_cove3[veg_cover3 == 0] = np.nan
    # FIG2 = plt.scatter(x, y, c=veg_cove3[-1, :], cmap='Reds', edgecolors='k', lw=0.2)
    # cbar = plt.colorbar(FIG2, label="Fraction cover Elytrigia [-]")
    # plt.title("Vegetation cover (Elytrigia, Spartina and Puccinellia) and Bed Level")
    # plt.xlabel("Grid cell x-direction")
    # plt.ylabel("Grid cell y-direction")
    # plt.show()

def plot_bl_diff(x,y,bl, veg_cover):
# # BED LEVEL DIFFERENCE (sedimentation/ erosion)
    bl_diff = bl[-1, :] - bl[0, :]
    fig = plt.figure(figsize=(12, 8))
    fig2 = plt.tricontourf(x, y, bl_diff , cmap = "terrain", levels=np.linspace(-1, 1, 80))
    cbar2 = plt.colorbar(fig2, label="Bed level difference [m]")
    plt.title("Bed Level Difference [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    plt.show()

    veg_cove = veg_cover.copy()
    veg_cove[veg_cover == 0] = np.nan
    FIG2 = plt.scatter(x, y, c=veg_cove[-1, :], cmap='Greens', edgecolors='k', lw = 0.2)
    cbar = plt.colorbar(FIG2, label="Fraction cover [-]")
    plt.title("Vegetation cover (Spartina) and Bed Level Difference [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    plt.show()


def plot_max_vel(x, y, max_u):
    fig = plt.figure(figsize=(12, 8))
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

def plot_cover_time(time, veg_cover, veg_dia):

    # fig, ax = plt.subplots()
    sum_cover = np.sum(veg_cover*((veg_dia/2)**2)*np.pi, axis = 1)
    sum_cover[np.isnan(sum_cover)== True] = 0.0
    time = pd.to_datetime(time, format='%Y%m%d').strftime('%d/%m/%Y')
    fig = plt.figure(figsize=(12, 8))

    plt.plot(time, sum_cover)
    axs = plt.gca()
    plt.title("Total Vegetation Cover")
    plt.xlabel("Time")
    plt.ylabel("Cover [m^2]")

    # axs.xaxis.set_minor_locator(mdates.MonthLocator(interval=12))
    # axs.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    # plt.gcf().autofmt_xdate()
    # axs.set_xticks(np.arange(0,504,24))
    plt.xticks(rotation=45)

    # mapfile2 = nc.Dataset(
    #     r"p:\11205209-rejuvenationzuidgors\Zuidgors_modelling\Zuidgors_SSC0.02_2boundaries_refinedGrid\output\\VegModel_Puccinellia_map.nc",
    #     "r",
    # )
    # veg_cover2 = ma.MaskedArray.filled((mapfile2.variables["cover"][:, :]), 0.0)
    # veg_dia2 = ma.MaskedArray.filled((mapfile2.variables["diaveg"][:, :]), 0.0)  # average stem diameter each cell
    #
    # sum_cover = np.sum(veg_cover2*((veg_dia2/2)**2)*np.pi, axis=1)
    # fig = plt.plot(time, sum_cover)
    # plt.title("Total Vegetation Cover")
    # plt.xlabel("Time")
    # plt.ylabel("Cover [m^2]")
    # plt.xticks(rotation=45)
    # # ax.legend(["Spartina", "Puccinellia"])
    #
    # mapfile3 = nc.Dataset(
    #     r"p:\11205209-rejuvenationzuidgors\Zuidgors_modelling\Zuidgors_ref\output_Elytrigia\\VegModel_Elytrigia_map.nc",
    #     "r",
    # )
    # veg_cover3 = ma.MaskedArray.filled((mapfile3.variables["cover"][:, :]), 0.0)
    # veg_dia3 = ma.MaskedArray.filled((mapfile3.variables["diaveg"][:, :]), 0.0)  # average stem diameter each cell
    #
    # sum_cover = np.sum(veg_cover3*((veg_dia3/2)**2)*np.pi, axis=1)
    # fig = plt.plot(time, sum_cover)
    # plt.title("Total Vegetation Cover")
    # plt.xlabel("Time")
    # plt.ylabel("Cover [m^2]")
    # plt.xticks(rotation=45)
    # plt.legend(["Spartina", "Puccinellia", "Elytrigia"])
    #
    # mapfile4 = nc.Dataset(
    #     r"p:\11205209-rejuvenationzuidgors\Zuidgors_modelling\Zuidgors_SSC0.02_2boundaries_refinedGrid\output\\VegModel_Salicornia_map.nc",
    #     "r",
    # )
    # veg_cover4 = ma.MaskedArray.filled((mapfile4.variables["cover"][:, :]), 0.0)
    # veg_dia4 = ma.MaskedArray.filled((mapfile4.variables["diaveg"][:, :]), 0.0)  # average stem diameter each cell
    #
    # sum_cover = np.sum(veg_cover4*((veg_dia4/2)**2)*np.pi, axis=1)
    # fig = plt.plot(time, sum_cover)
    # plt.title("Totoal Vegetation Cover")
    # plt.xlabel("Time")
    # plt.ylabel("Cover [m^2]")
    # plt.xticks(rotation=45)
    # plt.legend(["Spartina","Puccinellia", "Elytrigia", "Salicornia"])

mapfile = nc.Dataset(
    r"p:\11205209-rejuvenationzuidgors\Zuidgors_modelling\Zuidgors_bigger\output_1species\VegModel_Spartina_map.nc",
    "r",
)
bl, max_u, veg_den, veg_dia, veg_cover, time, veg_height, x, y, veg_frac_m, veg_frac_j, veg_frac_m, height_j, height_m, dia_m, dia_j, age_j, age_m = get_variables_mapfile(mapfile)

plot_cover_bath(x, y, bl, veg_cover)
# plot_veg_den(x, y, veg_den)
# plot_veg_height(x, y, veg_height)
# plot_bl_diff(x, y, bl, veg_cover)
# plot_max_vel(x, y, max_u)
# plot_cover_time(time, veg_cover, veg_dia)
# make_gif()
mapfile.close


# hisfile = nc.Dataset(r'c:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\Zuidgors_bigger\output\VegModel_Spartina_his.nc', 'r')
#
# diaveg, time, veg_frac, veg_den, veg_height, cover = get_variables_hisfile(hisfile)
# plot_cover_time(time, cover, 641)
#
# hisfile.close
#
