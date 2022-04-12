import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

# plt.style.use('seaborn-whitegrid')


out_dir = os.path.join(
    "C:\\",
    "Users",
    "toledoal",
    "NBSDynamics",
    "test",
    "test_data",
    "sm_testcase2",
    "input",
    "MinFiles",
    "fm",
    "output",
)

# read map file and plot
url = os.path.join(out_dir, "VegModel_map.nc")
nc = netCDF4.Dataset(url)
nc.variables.keys()
limdict = {
    "cover": [0, 9999],
    "height": [300, 304],
    "diaveg": [295, 297],
    "rnveg": [302, 304],
    "veg_frac_j": [0, 9999],
    "veg_frac_m": [0, 1.05],
    "max_tau": [0, 1.05],
    "max_u": [0, 1.05],
    "max_wl": [0, 1.05],
    "min_wl": [0, 1.05],
    "bl": [9999, 9999],
}

teller = 0
for vv in nc.variables.keys():
    teller = teller + 1
    if teller > 3:

        VT = nc.variables[vv]
        VarT = VT[:]

        fig = plt.figure()
        ax = plt.axes()
        plt.xlim(0, 100)
        ylims = limdict[vv]
        if ylims[0] == 9999:
            ylims[0] = 0.95 * np.min(VarT)
        if ylims[1] == 9999:
            ylims[1] = 1.05 * np.max(VarT)
        plt.ylim(ylims)
        plt.title(VT.long_name)
        plt.xlabel("Time (years)")
        plt.ylabel(VT.units)

        x = np.linspace(1, 100, 100)

        ax.plot(x, VarT[:, 1], "-g", label="Cell 1")
        ax.plot(x, VarT[:, 100], "-r", label="Cell 100")
        ax.plot(x, VarT[:, 300], "-c", label="Cell 300")
        plt.legend()

nc.close()

# read his file and plot
url = os.path.join(out_dir, "VegModel_his.nc")
nc = netCDF4.Dataset(url)
nc.variables.keys()

teller = 0
for vv in nc.variables.keys():
    teller = teller + 1
    if teller > 3:

        VT = nc.variables[vv]
        VarT = VT[:]

        fig = plt.figure()
        ax = plt.axes()
        plt.xlim(0, 100)
        ylims = limdict[vv]
        if ylims[0] == 9999:
            ylims[0] = 0.95 * np.min(VarT)
        if ylims[1] == 9999:
            ylims[1] = 1.05 * np.max(VarT)
        plt.title(VT.long_name)
        plt.xlabel("Time (years)")
        plt.ylabel(VT.units)

        x = np.linspace(0, 100, 36525)
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, VarT.shape[1])))
        for i in range(VarT.shape[1]):
            ax.plot(x, VarT[:, i], color=next(colors), label=f"Point {i}")
        plt.legend()

nc.close()
