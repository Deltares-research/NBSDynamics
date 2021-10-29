# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:36:48 2021

@author: herman
"""

import os
from typing import Callable

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

plt.style.use("seaborn-whitegrid")


def subplot_mapfile(var_t, ylims, plot_axes):
    x = np.linspace(1, 100, 100)
    plt.ylim(ylims)
    plot_axes.plot(x, var_t[:, 1], "-g", label="Cell 1")
    plot_axes.plot(x, var_t[:, 100], "-r", label="Cell 100")
    plot_axes.plot(x, var_t[:, 300], "-c", label="Cell 300")
    plt.legend()


def subplot_hisfile(var_t, ylims, plot_axes):
    x = np.linspace(0, 100, 36525)
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, var_t.shape[1])))
    for i in range(var_t.shape[1]):
        plot_axes.plot(x, var_t[:, i], color=next(colors), label=f"Point {i}")
    plt.legend()


def plot_nc_variables(nc_variables, subplot_call: Callable):
    teller = 0
    for vv in nc_variables.keys():
        teller = teller + 1
        if teller > 3:

            VT = nc_variables[vv]
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
            subplot_call(VarT, VT, ax)


# NetCDF4-Python can read a remote OPeNDAP dataset or a local NetCDF file:
out_dir = os.path.join(
    "C:\\",
    "Users",
    "herman",
    "OneDrive - Stichting Deltares",
    "Documents",
    "PyProjects",
    "Mariya_model",
    "Run_Transect",
    "output",
)

# read map file and plot
url = os.path.join(out_dir, "CoralModel_map.nc")
nc = netCDF4.Dataset(url)
nc.variables.keys()
limdict = {
    "Iz": [0, 9999],
    "Tc": [300, 304],
    "Tlo": [295, 297],
    "Thi": [302, 304],
    "PD": [0, 9999],
    "PT": [0, 1.05],
    "PH": [0, 1.05],
    "PR": [0, 1.05],
    "PP": [0, 1.05],
    "PB": [0, 1.05],
    "calc": [9999, 9999],
    "dc": [9999, 9999],
    "hc": [9999, 9999],
    "bc": [9999, 9999],
    "tc": [9999, 9999],
    "ac": [9999, 9999],
    "Vc": [9999, 9999],
    "G": [9999, 9999],
}

plot_nc_variables(nc.variables, subplot_mapfile)

nc.close()

# read his file and plot
url = os.path.join(out_dir, "CoralModel_his.nc")
nc = netCDF4.Dataset(url)
nc.variables.keys()
plot_nc_variables(nc.variables, subplot_hisfile)

nc.close()
