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
from datetime import datetime

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
    return bl, max_u, veg_den,veg_dia, veg_cover, time, veg_height, x, y, veg_frac_m, veg_frac_j, veg_frac_m, height_j, height_m, dia_m, dia_j, age_j, age_m

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
    fig = plt.tricontourf(x, y, bl[24, :] , cmap = "terrain", levels=np.linspace(0,4,80))
    cbar = plt.colorbar(fig, label="Bed level [m]")
    # plt.title("Vegetation cover (Spartina) and Bed Level [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")

    veg_cove = veg_cover.copy()
    veg_cove[veg_cover==0] = np.nan
    FIG2  = plt.scatter(x, y, c=veg_cove[24, :], cmap='Greens', edgecolors='k', lw = 0.2)
    cbar = plt.colorbar(FIG2,label="Fraction cover Spartina [-]")
    plt.title("Vegetation cover (Spartina) and Bed Level [m]")
    plt.xlabel("Grid cell x-direction")
    plt.ylabel("Grid cell y-direction")
    plt.show()

    # mapfile2 = nc.Dataset(
    #     r"p:\11205209-rejuvenationzuidgors\Zuidgors_modelling\Zuidgors_bigger\output_Spart_Pucci\\VegModel_Puccinellia_map.nc",
    #     "r",
    # )
    # veg_cover2 = ma.MaskedArray.filled((mapfile2.variables["cover"][:, :]), 0.0)
    # veg_cove2 = veg_cover2.copy()
    # veg_cove2[veg_cover2 == 0] = np.nan
    # FIG2 = plt.scatter(x, y, c=veg_cove2[-1, :], cmap='Blues', edgecolors='k', lw=0.2)
    # cbar = plt.colorbar(FIG2, label="Fraction cover Puccinellia [-]")
    # plt.title("Vegetation cover (Elytrigia) and Bed Level")
    # plt.xlabel("Grid cell x-direction")
    # plt.ylabel("Grid cell y-direction")
    # plt.show()

    # # #
    # mapfile3 = nc.Dataset(
    #     r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Design_scenarios\no_waves\Zuidgors_exArea_front\output\\VegModel_Elytrigia_map.nc",
    #     "r",
    # )
    # veg_cover3 = ma.MaskedArray.filled((mapfile3.variables["cover"][:, :]), 0.0)
    # veg_cove3 = veg_cover3.copy()
    # veg_cove3[veg_cover3 == 0] = np.nan
    # FIG2 = plt.scatter(x, y, c=veg_cove3[-1, :], cmap='Reds', edgecolors='k', lw=0.2)
    # cbar = plt.colorbar(FIG2, label="Fraction cover Elytrigia [-]")
    # plt.title("Vegetation cover (Elytrigia) and Bed Level")
    # plt.xlabel("Grid cell x-direction")
    # plt.ylabel("Grid cell y-direction")
    # plt.show()
    # #
    # mapfile4 = nc.Dataset(
    #     r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Design_scenarios\no_waves\Zuidgors_exArea_front\output\VegModel_Salicornia_map.nc",
    #     "r",
    # )
    # veg_cover4 = ma.MaskedArray.filled((mapfile4.variables["cover"][:, :]), 0.0)
    # veg_cove4 = veg_cover4.copy()
    # veg_cove4[veg_cover4 == 0] = np.nan
    # FIG2 = plt.scatter(x, y, c=veg_cove4[-1, :], cmap='Purples', edgecolors='k', lw=0.2)
    # cbar = plt.colorbar(FIG2, label="Fraction cover Salicornia [-]")
    # plt.title("Vegetation cover and Bed Level")
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

    veg_cove = np.zeros(veg_cover.shape)
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
    time = pd.to_datetime(time, format='%Y%m%d').strftime('%d/%m/%Y')
    fig = plt.figure(figsize=(12, 8))
    fig = plt.plot(time, sum_cover)
    plt.title("Total Vegetation Cover")
    plt.xlabel("Time")
    plt.ylabel("Cover [m^2]")
    plt.xticks(rotation=45)
    # plt.legend(["Salicornia"])

    # mapfile2 = nc.Dataset(
    #     r"p:\11205209-rejuvenationzuidgors\Zuidgors_modelling\Zuidgors_bigger\output_Spart_Pucci\\VegModel_Puccinellia_map.nc",
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
    # plt.legend(["Spartina", "Puccinellia"])
    #
    mapfile3 = nc.Dataset(
        r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Design_scenarios\no_waves\Zuidgors_exArea_front\output\\VegModel_Elytrigia_map.nc",
        "r",
    )
    veg_cover3 = ma.MaskedArray.filled((mapfile3.variables["cover"][:, :]), 0.0)
    veg_dia3 = ma.MaskedArray.filled((mapfile3.variables["diaveg"][:, :]), 0.0)  # average stem diameter each cell

    sum_cover = np.sum(veg_cover3*((veg_dia3/2)**2)*np.pi, axis=1)
    fig = plt.plot(time, sum_cover)
    plt.title("Total Vegetation Cover")
    plt.xlabel("Time")
    plt.ylabel("Cover [m^2]")
    plt.xticks(rotation=45)
    # plt.legend(["Spartina", "Elytrigia", "Puccinellia"])
    #
    mapfile4 = nc.Dataset(
        r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Design_scenarios\no_waves\Zuidgors_exArea_front\output\VegModel_Salicornia_map.nc",
        "r",
    )
    veg_cover4 = ma.MaskedArray.filled((mapfile4.variables["cover"][:, :]), 0.0)
    veg_dia4 = ma.MaskedArray.filled((mapfile4.variables["diaveg"][:, :]), 0.0)  # average stem diameter each cell

    sum_cover = np.sum(veg_cover4*((veg_dia4/2)**2)*np.pi, axis=1)
    fig = plt.plot(time, sum_cover)
    plt.title("Total Vegetation Cover")
    plt.xlabel("Time")
    plt.ylabel("Cover [m^2]")
    plt.xticks(rotation=45)
    plt.legend(["Spartina","Elytrigia", "Salicornia"])

def plot_cover_time_EX(time, veg_cover, veg_dia):

    # fig, ax = plt.subplots()
    sum_cover = np.sum(veg_cover[:, exArea]*((veg_dia[:, exArea]/2)**2)*np.pi, axis = 1)
    time = pd.to_datetime(time, format='%Y%m%d').strftime('%d/%m/%Y')
    fig = plt.figure(figsize=(12, 8))
    fig = plt.plot(time, sum_cover)
    plt.title("Vegetation Cover Excavated Area")
    plt.xlabel("Time")
    plt.ylabel("Cover [m^2]")
    plt.xticks(rotation=45)
    # plt.legend(["Salicornia"])

    # mapfile2 = nc.Dataset(
    #     r"p:\11205209-rejuvenationzuidgors\Zuidgors_modelling\Zuidgors_bigger\output_Spart_Pucci\VegModel_Puccinellia_map.nc",
    #     "r",
    # )
    # veg_cover2 = ma.MaskedArray.filled((mapfile2.variables["cover"][:, :]), 0.0)
    # veg_dia2 = ma.MaskedArray.filled((mapfile2.variables["diaveg"][:, :]), 0.0)  # average stem diameter each cell
    #
    # sum_cover = np.sum(veg_cover2[:, exArea]*((veg_dia2[:, exArea]/2)**2)*np.pi, axis=1)
    # fig = plt.plot(time, sum_cover)
    # plt.title("Vegetation Cover Excavated Area")
    # plt.xlabel("Time")
    # plt.ylabel("Cover [m^2]")
    # plt.xticks(rotation=45)
    # plt.legend(["Spartina", "Puccinellia"])

    mapfile3 = nc.Dataset(
        r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Design_scenarios\no_waves\Zuidgors_exArea_front\output\\VegModel_Elytrigia_map.nc",
        "r",
    )
    veg_cover3 = ma.MaskedArray.filled((mapfile3.variables["cover"][:, :]), 0.0)
    veg_dia3 = ma.MaskedArray.filled((mapfile3.variables["diaveg"][:, :]), 0.0)  # average stem diameter each cell

    sum_cover = np.sum(veg_cover3[:, exArea]*((veg_dia3[:, exArea]/2)**2)*np.pi, axis=1)
    fig = plt.plot(time, sum_cover)
    plt.title("Total Vegetation Cover")
    plt.xlabel("Time")
    plt.ylabel("Cover [m^2]")
    plt.xticks(rotation=45)
    # plt.legend(["Spartina", "Elytrigia", "Puccinellia"])
    #
    mapfile4 = nc.Dataset(
        r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Design_scenarios\no_waves\Zuidgors_exArea_front\output\\VegModel_Salicornia_map.nc",
        "r",
    )
    veg_cover4 = ma.MaskedArray.filled((mapfile4.variables["cover"][:, :]), 0.0)
    veg_dia4 = ma.MaskedArray.filled((mapfile4.variables["diaveg"][:, :]), 0.0)  # average stem diameter each cell

    sum_cover = np.sum(veg_cover4[:, exArea]*((veg_dia4[:, exArea]/2)**2)*np.pi, axis=1)
    fig = plt.plot(time, sum_cover)
    plt.title("Vegetation Cover Excavated Area")
    plt.xlabel("Time")
    plt.ylabel("Cover [m^2]")
    plt.xticks(rotation=45)
    plt.legend(["Spartina","Elytrigia", "Salicornia"])

def plt_bl_time_EX(time, bl):
    mean_bl = np.mean(bl[:, exArea], axis=1)
    time = pd.to_datetime(time, format='%Y%m%d').strftime('%d/%m/%Y')
    fig = plt.figure(figsize=(12, 8))
    fig = plt.plot(time, mean_bl)
    plt.title("Mean Bed Level Excavated Area")
    plt.xlabel("Time")
    plt.ylabel("Bed Level [m]")
    plt.xticks(rotation=45)

def plot_maxU_time_EX(time, max_u):
    mean_maxU = np.mean(max_u[:, exArea], axis=1)
    time = pd.to_datetime(time, format='%Y%m%d').strftime('%d/%m/%Y')
    fig = plt.figure(figsize=(12, 8))
    fig = plt.plot(time, mean_maxU)
    plt.title("Average Maximum Velocity Excavated Area")
    plt.xlabel("Time")
    plt.ylabel("Flow Velocity [m/s]")
    plt.xticks(rotation=45)

mapfile = nc.Dataset(
    r"p:\11205209-rejuvenationzuidgors\Zuidgors_modelling\Zuidgors_bigger\output_1species\VegModel_Spartina_map.nc",
    "r",
)
bl, max_u, veg_den, veg_dia, veg_cover, time, veg_height, x, y, veg_frac_m, veg_frac_j, veg_frac_m, height_j, height_m, dia_m, dia_j, age_j, age_m = get_variables_mapfile(mapfile)
plot_cover_bath(x,y,bl,veg_cover)

# exArea = [3788, 3867, 3946, 3947, 4024, 4025, 4026, 4103, 4104, 4105, 4106,
#        4182, 4183, 4184, 4185, 4260, 4261, 4262, 4263, 4264, 4336, 4337,
#        4338, 4339, 4340, 4341, 4412, 4413, 4414, 4415, 4416, 4417, 4418,
#        4488, 4489, 4490, 4491, 4492, 4493, 4562, 4563, 4564, 4565, 4566,
#        4567, 4568, 4636, 4637, 4638, 4639, 4640, 4641, 4708, 4709, 4710,
#        4711, 4712, 4713, 4714, 4780, 4781, 4782, 4783, 4784, 4785, 4850,
#        4851, 4852, 4853, 4854, 4855, 4920, 4921, 4922, 4923, 4924, 4925,
#        4988, 4989, 4990, 4991, 4992, 4993, 5055, 5056, 5057, 5058, 5059,
#        5060, 5061, 5122, 5123, 5124, 5125, 5126, 5127, 5187, 5188, 5189,
#        5190, 5191, 5192, 5193, 5252, 5253, 5254, 5255, 5256, 5257, 5315,
#        5316, 5317, 5318, 5319, 5320, 5321, 5378, 5379, 5380, 5381, 5382,
#        5383, 5439, 5440, 5441, 5442, 5443, 5444, 5445, 5500, 5501, 5502,
#        5503, 5504, 5505, 5559, 5560, 5561, 5562, 5563, 5564, 5565, 5618,
#        5619, 5620, 5621, 5622, 5623, 5675, 5676, 5677, 5678, 5679, 5680,
#        5732, 5733, 5734, 5735, 5736, 5788, 5789, 5790, 5791, 5843, 5844,
#        5845, 5897, 5898]

exArea = [3779, 3780, 3781, 3857, 3858, 3859, 3860, 3861, 3936, 3937, 3938,
       3939, 3940, 4015, 4016, 4017, 4018, 4019, 4020, 4094, 4095, 4096,
       4097, 4098, 4099, 4173, 4174, 4175, 4176, 4177, 4178, 4251, 4252,
       4253, 4254, 4255, 4256, 4257, 4328, 4329, 4330, 4331, 4332, 4333,
       4334, 4404, 4405, 4406, 4407, 4408, 4409, 4410, 4479, 4480, 4481,
       4482, 4483, 4484, 4485, 4486, 4553, 4554, 4555, 4556, 4557, 4558,
       4559, 4560, 4626, 4627, 4628, 4629, 4630, 4631, 4632, 4633, 4634,
       4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706, 4770, 4771, 4772,
       4773, 4774, 4775, 4776, 4777, 4778, 4840, 4841, 4842, 4843, 4844,
       4845, 4846, 4847, 4848, 4909, 4910, 4911, 4912, 4913, 4914, 4915,
       4916, 4917, 4978, 4979, 4980, 4981, 4982, 4983, 4984, 4985, 4986,
       5045, 5046, 5047, 5048, 5049, 5050, 5051, 5052, 5053, 5111, 5112,
       5113, 5114, 5115, 5116, 5117, 5118, 5119, 5120, 5176, 5177, 5178,
       5179, 5180, 5181, 5182, 5183, 5184, 5185, 5241, 5242, 5243, 5244,
       5245, 5246, 5247, 5248, 5249, 5250, 5307, 5308, 5309, 5310, 5311,
       5312, 5313, 5373, 5374, 5375, 5376, 5437]


# plot_veg_den(x, y, veg_den)
# plot_veg_height(x, y, veg_height)
# plot_bl_diff(x, y, bl, veg_cover)
# plot_max_vel(x, y, max_u)
# plot_cover_time(time, veg_cover, veg_dia)
# make_gif()
mapfile.close

#
# hisfile = nc.Dataset(r'c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Zuidgors_ref\output\VegModel_Spartina_his.nc', 'r')
#
# diaveg, time, veg_frac, veg_den, veg_height, cover = get_variables_hisfile(hisfile)
# plot_cover_time(time, cover, 641)
#
# hisfile.close

