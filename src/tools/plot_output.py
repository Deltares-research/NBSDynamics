# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:36:48 2021

@author: herman
"""

import platform
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
from netCDF4 import Dataset

from src.core.output_model import Output

platform_sys = platform.system().lower()
if platform_sys in ["windows"]:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-whitegrid")
else:

    matplotlib.use("Agg")
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


class OutputPlot:
    def _plot_nc_variables(self, nc_variables: dict, subplot_call: Callable):
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

    def plot_map(self, map_path: Path):
        """
        Plots the map file according to the coral model.

        Args:
            map_path (Path): Path to the netcdf file representing the map.
        """

        def _subplot_mapfile(var_t: np.ma.MaskedArray, ylims: list, plot_axes):
            x = np.linspace(1, 100, 100)
            plt.ylim(ylims)
            plot_axes.plot(x, var_t[:, 1], "-g", label="Cell 1")
            plot_axes.plot(x, var_t[:, 100], "-r", label="Cell 100")
            plot_axes.plot(x, var_t[:, 300], "-c", label="Cell 300")
            plt.legend()

        with Dataset(map_path) as nc:
            self._plot_nc_variables(nc.variables, _subplot_mapfile)

    def plot_his(self, his_path: Path):
        """
        Plots the his file according to the coral model.

        Args:
            his_path (Path): Path to the netcdf file representing the his.
        """

        def _subplot_hisfile(var_t: np.ma.MaskedArray, ylims: list, plot_axes):
            x = np.linspace(0, 100, 36525)
            colors = iter(plt.cm.rainbow(np.linspace(0, 1, var_t.shape[1])))
            plt.ylim(ylims)
            for i in range(var_t.shape[1]):
                plot_axes.plot(x, var_t[:, i], color=next(colors), label=f"Point {i}")
            plt.legend()

        with Dataset(his_path) as nc:
            self._plot_nc_variables(nc.variables, _subplot_hisfile)


def plot_output(output_model: Output):
    """
    Plots the map and his files from an output model.

    Args:
        output_model (Output): Output model containing map and his files.

    Raises:
        ValueError: When no input argument has been provided.
    """
    if not isinstance(output_model, Output):
        raise ValueError("No output model provided.")
    output_plot = OutputPlot()
    output_plot.plot_map(output_model.file_name_map)
    output_plot.plot_his(output_model.file_name_his)
