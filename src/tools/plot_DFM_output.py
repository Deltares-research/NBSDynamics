import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib as mpl



mapfile = nc.Dataset(
    r"c:\Users\dzimball\PycharmProjects\NBSDynamics_Current\test\test_data\Design_scenarios\no_waves\Zuidgors_exArea_front\input\output\\Zuidgors_bigger_map.nc",
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



