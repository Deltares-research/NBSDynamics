# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:36:48 2021

@author: herman
"""

import platform
from pathlib import Path
from typing import Callable

import numpy as np
from netCDF4 import Dataset

if "windows" in platform.system().lower():
    import matplotlib.pyplot as plt


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


def init_matplotlib():
    plt.style.use("seaborn-whitegrid")


def _plot_nc_variables(nc_variables, subplot_call: Callable):
    teller = 0
    for vv in nc_variables.keys():
        teller = teller + 1
        if teller > 3:

            VT = nc_variables[vv]
            VarT = VT[:]

            plt.figure()
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
            subplot_call(VarT, ylims, ax)
            plt.close()


# read map file and plot
def plot_map(map_path: Path):
    """
    Plots the map file according to the coral model.

    Args:
        map_path (Path): Path to the netcdf file representing the map.
    """
    if "win" not in platform.system().lower():
        return
    init_matplotlib()

    def _subplot_mapfile(var_t, ylims, plot_axes):
        x = np.linspace(1, 100, 100)
        plt.ylim(ylims)
        plot_axes.plot(x, var_t[:, 1], "-g", label="Cell 1")
        plot_axes.plot(x, var_t[:, 100], "-r", label="Cell 100")
        plot_axes.plot(x, var_t[:, 300], "-c", label="Cell 300")
        plt.legend()

    with Dataset(map_path) as nc:
        _plot_nc_variables(nc.variables, _subplot_mapfile)


# read his file and plot
def plot_his(his_path: Path):
    """
    Plots the his file according to the coral model.

    Args:
        his_path (Path): Path to the netcdf file representing the his.
    """
    if "win" not in platform.system().lower():
        return
    init_matplotlib()

    def _subplot_hisfile(var_t, ylims, plot_axes):
        x = np.linspace(0, 100, 36525)
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, var_t.shape[1])))
        for i in range(var_t.shape[1]):
            plot_axes.plot(x, var_t[:, i], color=next(colors), label=f"Point {i}")
        plt.legend()

    with Dataset(his_path) as nc:
        _plot_nc_variables(nc.variables, _subplot_hisfile)
