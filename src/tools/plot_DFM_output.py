import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib as mpl



mapfile = nc.Dataset(
    r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Zuidgors_bigger_ref\input\output\Zuidgors_bigger_ref_map.nc",
    "r",
)

time = ma.MaskedArray.filled((mapfile.variables["time"][:]), 0.0)/3600/24
timestep = ma.MaskedArray.filled((mapfile.variables["timestep"][:]), 0.0)
current_morft = ma.MaskedArray.filled((mapfile.variables["morft"][:]), 0.0)/3600/24
x = ma.MaskedArray.filled((mapfile.variables["mesh2d_face_x"][:]), 0.0)
y = ma.MaskedArray.filled((mapfile.variables["mesh2d_face_y"][:]), 0.0)
vel_mag = ma.MaskedArray.filled((mapfile.variables["mesh2d_ucmag"][:, :]), 0.0)
water_depth = ma.MaskedArray.filled((mapfile.variables["mesh2d_waterdepth"][:, :]), 0.0)
# bed_level = ma.MaskedArray.filled((mapfile.variables["mesh2d_flowelem_bl"][:]), 0.0)
bed_level = mapfile.variables["mesh2d_mor_bl"][:, :]
# taumax = mapfile.variables["mesh2d_tausmax"]

fig = plt.figure(figsize=(12, 8))
fig1 = plt.tricontourf(x, y, water_depth[-1, :])
cbar = plt.colorbar(fig1, label="depth [m]")
plt.title("Water Depth")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.show()


# fig2, ax = plt.subplots()
fig = plt.figure(figsize=(12, 8))
fig = plt.tricontourf(x, y, vel_mag[-1, :])
cbar = plt.colorbar(fig, label="Flow velocity [m/s]")
plt.title("Flow Velocity Magnitude")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.show()

# # fig3, ax = plt.subplots()
# # # ax.plot(time, vel_mag[:, 324], label='Boundary1')
# # # ax.plot(time, vel_mag[:, 349], label='After Dam')
# # # ax.plot(time, vel_mag[:, 290], label='Mid Domain')
# fig = plt.figure(figsize=(12, 8))
# plt.plot(time, vel_mag[:, 1225], label='Boundary1')
# plt.plot(time, vel_mag[:, 1276], label='After Dam1')
# plt.plot(time, vel_mag[:, 1300], label='Mid Domain')
# # # ax.plot(time, vel_mag[:, 151], label='High Erosion Cell')
# # # ax.plot(time, vel_mag[:, 169], label='High Erosion Cell')
# # # ax.plot(time, vel_mag[:, 187], label='High Erosion Cell')
# # # ax.plot(time, vel_mag[:, 188], label='High Erosion Cell')
# # # ax.plot(time, vel_mag[:, 208], label='High Erosion Cell')
# # # ax.plot(time, vel_mag[:, 228], label='High Erosion Cell')
# # # ax.plot(time, vel_mag[:, 229], label='High Erosion Cell')
# # # ax.plot(time, vel_mag[:, 250], label='High Erosion Cell')
# legend = plt.legend(loc='best', shadow=True)
# plt.title("Flow Velocity Magnitude Boundary1")
# plt.xlabel("Time [days]")
# plt.ylabel("Velocity [m/s]")
# plt.show()

discharge = vel_mag[0:360, 349]*3.14
dt = 1*3600
volume = discharge*dt

# fig7, ax = plt.subplots()
# ax.plot(time, vel_mag[:, 136], label='Boundary2')
# # ax.plot(time, vel_mag[:, 154], label='After Dam2')
fig = plt.figure(figsize=(12, 8))
# plt.plot(time, vel_mag[:, 594], label='Boundary2')
# plt.plot(time, vel_mag[:, 628], label='After Dam2')
plt.plot(time, vel_mag[:, 3867], label='Inflow1 Excavated Area')
plt.plot(time, vel_mag[:, 4011], label='Inflow2 Excavated Area')
legend = plt.legend(loc='best', shadow=True)
plt.title("Flow Velocity Magnitude")
plt.xlabel("Time [days]")
plt.ylabel("Velocity [m/s]")
plt.show()



bl_diff = bed_level[-1, :] - bed_level[0, :]
# fig4, ax = plt.subplots()

fig = plt.figure(figsize=(12, 8))
fig = plt.tricontourf(x, y, bl_diff, cmap="terrain", levels=np.linspace(-1, 1, 80))
cbar2 = plt.colorbar(fig, label="Bed level difference [m]")
plt.title("Bed Level Difference [m]")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.show()

# fig5, ax = plt.subplots()
fig = plt.figure(figsize=(12, 8))
fig = plt.tricontourf(x, y, bed_level[-1, :], cmap="terrain", levels=np.linspace(0, 4, 100))
cbar = plt.colorbar(fig, label="Bed level [m]")
plt.title("Bed Level")
plt.xlabel("Grid cell n-direction")
plt.ylabel("Grid cell m-direction")
plt.show()

# fig6, ax = plt.subplots()

fig = plt.figure(figsize=(12, 8))
# plt.plot(time, bed_level[:, 1225], label='Boundary1')
# plt.plot(time, bed_level[:, 1276], label='After Dam1')
# plt.plot(time, bed_level[:, 1300], label='Mid Domain')
# plt.plot(time, bed_level[:, 594], label='Boundary2')
# plt.plot(time, bed_level[:, 628], label='After Dam2')
plt.plot(time, bed_level[:, 3867], label='Inflow1 Excavated Area')
plt.plot(time, bed_level[:, 4011], label='Inflow2 Excavated Area')
legend = plt.legend(loc='best', shadow=True)
plt.title("Bed Level")
plt.xlabel("Time [days]")
plt.ylabel("Bed level [m]")
plt.show()


# mapfile.close()
#
#
# for varfm in ['mesh2d_waterdepth','mesh2d_mor_bl', 'mesh2d_ucmag', 'mesh2d_tausmax']:
# varfm = 'mesh2d_waterdepth'
# for i in range(370,1240,10):
#     var = mapfile.variables[varfm][i, :]
#     # var =  ma.MaskedArray.filled((mapfile.variables[varfm][i, :]), 0.0)
#     fig = plt.tricontourf(x, y, var,  levels=np.linspace(0, 2, 100))
#     plt.gca().set_aspect('equal')
#     plt.colorbar(fig, spacing='uniform')
#     # plt.clim(0,2)
#     figname = (r'c:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\Zuidgors_bigger\figures\\Z_{}_ts{}.png'.format(varfm,str(i)))
#     plt.savefig(figname,dpi=300,bbox_inches='tight')
#     plt.close()





#
# """
# Make GIF of domain over time
# """
# from celluloid import Camera  # getting the camera
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'
# import numpy as np
# from IPython import display
# from IPython.display import HTML  # to show the animation in Jupyter
# import ffmpeg
# from matplotlib.animation import PillowWriter
#
# plt.rcParams[
#     'animation.ffmpeg_path'] = r'C:\Users\dzimball\ffmpeg-tools-2022-01-01-git-d6b2357edd\ffmpeg-tools-2022-01-01-git-d6b2357edd\bin\ffmpeg.exe'
#
# fig, ax = plt.subplots()  # creating my fig
# camera = Camera(fig)  # the camera gets the fig we'll plot
# for i in range(1000, 3000, 10):
#     fig = plt.tricontourf(x, y, vel_mag[i, :])  # WHAT NEEDS TO BE PLOTTED
#
#     plt.title("Flow Velocity")
#     plt.xlabel("Grid cell n-direction")
#     plt.ylabel("Grid cell m-direction")
#     plt.gca().set_aspect('equal')
#     # plt.colorbar(fig, spacing='uniform')
#     camera.snap()  # the camera takes a snapshot of the plot
# animation = camera.animate(interval=800, repeat=True,
#                            repeat_delay=300)  # animation ready
# HTML(animation.to_html5_video())  # displaying the animation
#
# animation.save(
#     r'c:\Users\dzimball\Zuidgors_new\animation_vel.gif',
#     writer='PillowWriter', fps=2)

# cbar = plt.colorbar(fig, label="Velocity [m/s]")


x1 = [48364.5,
48344.5,
48324.5,
48304.5,
48284.5,
48264.5,
48244.5,
48224.5,
48204.5,
48184.5,
48164.5,
48144.5,
48124.5,
48104.5,
48084.5,
48064.5,
48044.5,
48024.5,
48004.5,
47984.5,
47964.5,
47944.5,
47924.5,
47904.5,
47884.5
]

y1 = [378967.7,
378977.7,
378987.7,
378997.7,
379007.7,
379017.7,
379027.7,
379037.7,
379047.7,
379057.7,
379067.7,
379077.7,
379087.7,
379097.7,
379107.7,
379117.7,
379127.7,
379137.7,
379147.7,
379157.7,
379167.7,
379177.7,
379187.7,
379197.7,
379207.7
]

x2 = [47814.5,
47834.5,
47854.5,
47874.5,
47894.5,
47914.5,
47934.5,
47954.5,
47974.5,
47994.5,
48014.5,
48034.5,
48054.5,
48074.5,
48094.5,
48114.5,
48134.5,
48154.5,
48174.5,
48194.5,
48214.5,
48234.5,
48254.5,
48274.5,
48294.5,
48314.5,
48334.5,
48354.5,
48374.5,
48394.5
]

y2 = [379287.7,
379277.7,
379267.7,
379257.7,
379247.7,
379237.7,
379227.7,
379217.7,
379207.7,
379197.7,
379187.7,
379177.7,
379167.7,
379157.7,
379147.7,
379137.7,
379127.7,
379117.7,
379107.7,
379097.7,
379087.7,
379077.7,
379067.7,
379057.7,
379047.7,
379037.7,
379027.7,
379017.7,
379007.7,
378997.7
]

x3 = [48294.5,
48274.5,
48254.5,
48234.5,
48214.5,
48194.5,
48174.5,
48154.5,
48134.5,
48114.5,
48094.5,
48074.5,
48054.5,
48034.5,
48014.5,
47994.5,
47974.5,
47954.5,
47934.5,
47914.5,
47894.5,
47874.5,
47854.5,
47834.5,
47814.5,
47794.5,
47774.5,
47754.5,
47734.5,
47714.5,
47694.5,
47674.5,
47654.5
]

y3 = [378907.7,
378917.7,
378927.7,
378937.7,
378947.7,
378957.7,
378967.7,
378977.7,
378987.7,
378997.7,
379007.7,
379017.7,
379027.7,
379037.7,
379047.7,
379057.7,
379067.7,
379077.7,
379087.7,
379097.7,
379107.7,
379117.7,
379127.7,
379137.7,
379147.7,
379157.7,
379167.7,
379177.7,
379187.7,
379197.7,
379207.7,
379217.7,
379227.7
]

x4 = [47694.5,
47714.5,
47734.5,
47754.5,
47774.5,
47794.5,
47814.5,
47834.5,
47854.5,
47874.5,
47894.5,
47914.5,
47934.5,
47954.5,
47974.5,
47994.5,
48014.5,
48034.5,
48054.5,
48074.5,
48094.5,
48114.5,
48134.5,
48154.5,
48174.5,
48194.5,
48214.5,
48234.5,
48254.5
]

y4= [379257.7,
379247.7,
379237.7,
379227.7,
379217.7,
379207.7,
379197.7,
379187.7,
379177.7,
379167.7,
379157.7,
379147.7,
379137.7,
379127.7,
379117.7,
379107.7,
379097.7,
379087.7,
379077.7,
379067.7,
379057.7,
379047.7,
379037.7,
379027.7,
379017.7,
379007.7,
378997.7,
378987.7,
378977.7
]

x5 = [48194.5,
48174.5,
48154.5,
48134.5,
48114.5,
48094.5,
48074.5,
48054.5,
48034.5,
48014.5,
47994.5,
47974.5,
47954.5,
47934.5,
47914.5,
47894.5,
47874.5,
47854.5,
47834.5,
47814.5,
47794.5,
47774.5,
47754.5,
47734.5,
47714.5,
47694.5,
47674.5,
47654.5,
47634.5
]

y5 = [378897.7,
378907.7,
378917.7,
378927.7,
378937.7,
378947.7,
378957.7,
378967.7,
378977.7,
378987.7,
378997.7,
379007.7,
379017.7,
379027.7,
379037.7,
379047.7,
379057.7,
379067.7,
379077.7,
379087.7,
379097.7,
379107.7,
379117.7,
379127.7,
379137.7,
379147.7,
379157.7,
379167.7,
379177.7
]

c1 = np.zeros(len(x1))
# xfind = np.zeros(len(x1))
# yfind = np.zeros(len(x1))
for i in range(0, len(x1)):
    xfind = np.where(x1[i] == x)
    yfind = np.where(y1[i] == y)
    a = set(xfind[0]).intersection(yfind[0])
    c1[i] = list(a)[0]

c2 = np.zeros(len(x2))
for i in range(0, len(x2)):
    xfind = np.where(x2[i] == x)
    yfind = np.where(y2[i] == y)
    a = set(xfind[0]).intersection(yfind[0])
    c2[i] = list(a)[0]

c3 = np.zeros(len(x3))
for i in range(0, len(x3)):
    xfind = np.where(x3[i] == x)
    yfind = np.where(y3[i] == y)
    a = set(xfind[0]).intersection(yfind[0])
    c3[i] = list(a)[0]

c4 = np.zeros(len(x4))
for i in range(0, len(x4)):
    xfind = np.where(x4[i] == x)
    yfind = np.where(y4[i] == y)
    a = set(xfind[0]).intersection(yfind[0])
    c4[i] = list(a)[0]

c5 = np.zeros(len(x5))
for i in range(0, len(x5)):
    xfind = np.where(x5[i] == x)
    yfind = np.where(y5[i] == y)
    a = set(xfind[0]).intersection(yfind[0])
    c5[i] = list(a)[0]

# ## Plot the cross sections of bed level
bl1 = np.zeros([len(c1), 3])
bl2 = np.zeros([len(c2), 3])
bl3 = np.zeros([len(c3), 3])
bl4 = np.zeros([len(c4), 3])
bl5 = np.zeros([len(c5), 3])
for y in range(0, 3):
    time = [0, 8800, 17600]
    for i in range(0, len(c1)):
        bl1[i, y] = bed_level[time[y], int(c1[i])]
    for i in range(0, len(c2)):
        bl2[i, y] = bed_level[time[y], int(c2[i])]
    for i in range(0, len(c3)):
        bl3[i, y] = bed_level[time[y], int(c3[i])]
    for i in range(0, len(c4)):
        bl4[i, y] = bed_level[time[y], int(c4[i])]
    for i in range(0, len(c5)):
        bl5[i, y] = bed_level[time[y], int(c5[i])]

fig, axs = plt.subplots(5)
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Bed level cross-sections')
# plt.title("Bed Level")
plt.xlabel("Distance")
# plt.legend(['Year 0', 'Year 1', 'Year 2'])
# plt.ylabel("Bed level [m]")
# plt.show()
axs[0].plot(bl1[:, 0])
axs[0].plot(bl1[:, 1])
axs[0].plot(bl1[:, 2])
axs[0].set_ylabel("Bed level [m]")
axs[0].legend(['Year 0', 'Year 1', 'Year 2'])
axs[1].plot(bl2[:, 0])
axs[1].plot(bl2[:, 1])
axs[1].plot(bl2[:, 2])
axs[1].set_ylabel("Bed level [m]")
# axs[1].legend(['Year 0', 'Year 1', 'Year 2'])
axs[2].plot(bl3[:, 0])
axs[2].plot(bl3[:, 1])
axs[2].plot(bl3[:, 2])
axs[2].set_ylabel("Bed level [m]")
# axs[2].legend(['Year 0', 'Year 1', 'Year 2'])
axs[3].plot(bl4[:, 0])
axs[3].plot(bl4[:, 1])
axs[3].plot(bl4[:, 2])
axs[3].set_ylabel("Bed level [m]")
# axs[3].legend(['Year 0', 'Year 1', 'Year 2'])
axs[4].plot(bl5[:, 0])
axs[4].plot(bl5[:, 1])
axs[4].plot(bl5[:, 2])
axs[4].set_ylabel("Bed level [m]")
# axs[4].legend(['Year 0', 'Year 1', 'Year 2'])

fig.show()




