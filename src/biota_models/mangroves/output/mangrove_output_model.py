import os.path
from datetime import datetime
from typing import Optional, Union

import numpy as np
from netCDF4 import Dataset
from pandas import DataFrame
from tkinter import messagebox
import os.path
from os import path
import sys

from src.biota_models.mangroves.model.mangrove_model import Mangrove
from src.core.output.base_output_model import BaseOutput, BaseOutputParameters


class MangroveOutputParameters(BaseOutputParameters):
    hydro_mor: bool = True  # hydro_morph micro-environment
    mangrove_characteristics: bool = True  # mangrove characteristics


class _MangroveOutput(BaseOutput):
    """
    Base class containing the generic definition of a 'Mangrove' output model.
    """

    # Output model attributes.
    output_params: MangroveOutputParameters = MangroveOutputParameters()


class MangroveMapOutput(_MangroveOutput):
    """
    Object representing a Map output. Implements the 'OutputProtocol'.
    """

    output_filename = "MangroveModel_map.nc"
    xy_coordinates: Optional[np.ndarray]
    first_year: Optional[int]

    @property
    def space(self) -> int:
        """
        Gets the total space for the model-defined xy-coordinates.

        Returns:
            int: length of 'xy_coordinates'.
        """
        return len(self.xy_coordinates)

    def initialize(self, mangrove: Optional[Mangrove]):
        self.output_filename = "MangroveModel_map.nc"
        """Initiate mapping output file in which output covering the whole model domain is stored every period of running."""
        if not self.valid_output():
            return
        if os.path.exists(self.output_filepath):
            askokcancel = messagebox.askokcancel("File overwrite", 'Do you want to overwrite existing output files?')
            if askokcancel == False:
                sys.exit()
        # Open netcdf data and initialize needed variables.
        with Dataset(self.output_filepath, "w", format="NETCDF4") as _map_data:
            _map_data.description = "Mapped simulation data of the MangroveModel."

            # dimensions
            _map_data.createDimension("time", None)
            _map_data.createDimension("nmesh2d_face", self.space)
            _map_data.createDimension("age", None)

            # variables
            t = _map_data.createVariable("time", int, ("time",))
            t.long_name = "period"
            t.units = "period"

            age = _map_data.createVariable("age", int, ("age"))
            age.long_name = "mangrove age"
            age.units = "ets"

            x = _map_data.createVariable("nmesh2d_x", "f8", ("nmesh2d_face",))
            x.long_name = "x-coordinate"
            x.units = "m"

            y = _map_data.createVariable("nmesh2d_y", "f8", ("nmesh2d_face",))
            y.long_name = "y-coordinate"
            y.units = "m"

            # t[:] = self.first_year
            x[:] = self.xy_coordinates[:, 0]
            y[:] = self.xy_coordinates[:, 1]


            def init_hydro_mor():
                max_tau = _map_data.createVariable(
                    "max_tau", "f8", ("time", "nmesh2d_face")
                )
                max_tau.long_name = "maximum bed shear stress"
                max_tau.units = "N/m^2"
                max_tau[:, :] = 0
                max_u = _map_data.createVariable(
                    "max_u", "f8", ("time", "nmesh2d_face")
                )
                max_u.long_name = "maximum flow velocity"
                max_u.units = "m/s"
                max_u[:, :] = 0
                max_wl = _map_data.createVariable(
                    "max_wl", "f8", ("time", "nmesh2d_face")
                )
                max_wl.long_name = "maximum water level"
                max_wl.units = "m"
                max_wl[:, :] = 0
                min_wl = _map_data.createVariable(
                    "min_wl", "f8", ("time", "nmesh2d_face")
                )
                min_wl.long_name = "minimum water level"
                min_wl.units = "m"
                min_wl[:, :] = 0
                bl = _map_data.createVariable("bl", "f8", ("time", "nmesh2d_face"))
                bl.long_name = "bedlevel"
                bl.units = "m"
                bl[:, :] = 0

            def init_mangrove_characteristics():

                height = _map_data.createVariable(
                    "height", "f8", ("nmesh2d_face", "age", "time")
                )
                height.long_name = "mangrove tree height"
                height.units = "m"
                height[:, :, :] = 0

                stem_dia = _map_data.createVariable(
                    "stem_dia", "f8",("nmesh2d_face", "age", "time")
                )
                stem_dia.long_name = "stem diameter"
                stem_dia.units = "cm"
                stem_dia[:, :, :] = 0

                stem_num = _map_data.createVariable(
                    "stem_num", "f8",("nmesh2d_face", "age", "time")
                )
                stem_num.long_name = "stem number per grid cell"
                stem_num.units = "-"
                stem_num[:, :, :] = 0

                root_num = _map_data.createVariable(
                    "root_num", "f8", ("nmesh2d_face", "age", "time")
                )
                root_num.long_name = "stem number per grid cell"
                root_num.units = "-"
                root_num[:, :, :] = 0

                tot_biomass = _map_data.createVariable(
                    "tot_biomass", "f8", ("time", "nmesh2d_face")
                )
                tot_biomass.long_name = "stem number per grid cell"
                tot_biomass.units = "-"
                tot_biomass[:, :] = 0


            conditions_funct = dict(
                hydro_mor=init_hydro_mor,
                mangrove_characteristics=init_mangrove_characteristics,
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()

    def update(self, mangrove: Mangrove, end_time: int, ets, year, constants):
        """Write data every period ran covering the whole model domain.

        :param mangrove: Mangrove
        :param period: based on mangrove_base_simulation

        :type mangrove: Mangrove
        :type period: int
        """
        if not self.valid_output():
            return
        with Dataset(self.output_filepath, mode="a") as _map_data:
            i = ets + constants.t_eco_year * year

            _map_data["time"][i] = end_time

            def update_hydro_mor():
                _map_data["max_tau"][-1, :] = mangrove.max_tau
                _map_data["max_u"][-1, :] = mangrove.max_u
                _map_data["max_wl"][-1, :] = mangrove.max_wl
                _map_data["min_wl"][-1, :] = mangrove.min_wl
                _map_data["bl"][-1, :] = mangrove.bl

            def update_mangrove_characteristics():
                _map_data["stem_num"][:, :, -1] = mangrove.stem_num[:, :]
                _map_data["stem_dia"][:, :, -1] = mangrove.stem_dia[:, :]
                _map_data["height"][:, :, -1] = mangrove.height[:, :]
                _map_data["root_num"][:, :, -1] = mangrove.root_num[:, :]
                _map_data["tot_biomass"][-1, :] = mangrove.bio_total_cell


            conditions_funct = dict(
                hydro_mor=update_hydro_mor,
                mangrove_characteristics=update_mangrove_characteristics,
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()


class MangroveHisOutput(_MangroveOutput):
    """
    Object representing a His output. Implements the 'OutputProtocol'.
    """

    output_filename = "MangroveModel_his.nc"
    xy_stations: Optional[np.ndarray]
    idx_stations: Optional[np.ndarray]
    first_date: Optional[Union[np.datetime64, datetime]]

    def initialize(self, mangrove: Mangrove):
        """Initiate history output file in which daily output at predefined locations within the model is stored."""
        self.output_filename = "MangroveModel_his.nc"
        if not self.valid_output():
            return
        if os.path.exists(self.output_filepath):
            askokcancel = messagebox.askokcancel("File overwrite", 'Do you want to overwrite existing output files?')
            if askokcancel == False:
                sys.exit()
        with Dataset(self.output_filepath, "w", format="NETCDF4") as _his_data:
            _his_data.description = "Historic simulation data of the MangroveModel"

            # dimensions
            _his_data.createDimension("time", None)
            _his_data.createDimension("stations", len(self.xy_stations))

            # variables
            t = _his_data.createVariable("time", "f8", ("time",))
            t.long_name = f"days since {self.first_date}"
            t.units = "days"

            x = _his_data.createVariable("station_x_coordinate", "f8", ("stations",))
            y = _his_data.createVariable("station_y_coordinate", "f8", ("stations",))

            # setup data set
            x[:] = self.xy_stations[:, 0]
            y[:] = self.xy_stations[:, 1]

            def init_hydro_mor():
                tau = _his_data.createVariable(
                    "tau", "f8", ("stations", "time")
                )
                tau.long_name = "maximum bed shear stress"
                tau.units = "N/m^2"
                tau[:, :] = 0
                u = _his_data.createVariable("u", "f8", ("stations", "time"))
                u.long_name = "flow velocity"
                u.units = "m/s"
                u[:, :] = 0
                wl = _his_data.createVariable("wl", "f8", ("stations", "time"))
                wl.long_name = "water level"
                wl.units = "m"
                wl[:, :] = 0
                bl = _his_data.createVariable("bl", "f8", ("stations", "time"))
                bl.long_name = "bedlevel"
                bl.units = "m"
                bl[:, :] = 0

            def init_mangrove_characteristics():

                height = _his_data.createVariable("height", "f8", ("stations", "time"))
                height.long_name = "mangrove tree height"
                height.units = "m"
                height[:, :] = 0

                stem_dia = _his_data.createVariable("stem_dia", "f8", ("stations", "time"))
                stem_dia.long_name = "stem diameter"
                stem_dia.units = "m"
                stem_dia[:, :] = 0

                stem_num = _his_data.createVariable("stem_num", "f8", ("stations", "time"))
                stem_num.long_name = "stem number per grid cell"
                stem_num.units = "-"
                stem_num[:, :] = 0

                root_num = _his_data.createVariable("root_num", "f8", ("stations", "time"))
                root_num.long_name = "root number per grid cell"
                root_num.units = "-"
                root_num[:, :] = 0

                tot_biomass = _his_data.createVariable("tot_biomass", "f8", ("stations", "time"))
                tot_biomass.long_name = "total biomass per grid cell"
                tot_biomass.units = "-"
                tot_biomass[:, :] = 0


            conditions_funct = dict(
                hydro_mor=init_hydro_mor,
                mangrove_characteristics=init_mangrove_characteristics,
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()

    def update(self, mangrove: Mangrove, dates: DataFrame):
        """Write data as daily output at predefined locations within the model domain.

        :param vegetation: Vegetation
        :param dates: dates of simulation year

        :type vegetation: Vegetation
        :type dates: DataFrame
        """
        if not self.valid_output():
            return
        with Dataset(self.output_filepath, mode="a") as _his_data:
            y_dates = dates.reset_index(drop=True)
            ti = ((y_dates - self.first_date).squeeze()).dt.days.values
            _his_data["time"][ti[:]] = y_dates.values

            def update_hydro_mor():
                _his_data["tau"][:, ti] = mangrove.tau_ts[:, 0:len(ti)]
                _his_data["u"][:, ti] = mangrove.u_ts[:, 0:len(ti)]
                _his_data["wl"][:, ti] = mangrove.wl_ts[:, 0:len(ti)]
                _his_data["bl"][:, ti] = mangrove.bl_ts[:, 0:len(ti)]

            def update_mangrove_characteristics():
                _his_data["stem_num"][:, ti] = np.tile(
                    np.mean(mangrove.stem_num, axis=1), (1, len(y_dates))
                )[self.idx_stations, :]
                _his_data["height"][:, ti] = np.tile(
                    np.mean(mangrove.height, axis=1), (1, len(y_dates))
                )[self.idx_stations, :]
                _his_data["stem_dia"][:, ti] = np.tile(
                    np.mean(mangrove.stem_dia, axis=1), (1, len(y_dates))
                )[self.idx_stations, :]
                _his_data["root_num"][:, ti] = np.tile(
                    np.mean(mangrove.root_num, axis=1), (1, len(y_dates))
                )[self.idx_stations, :]
                _his_data["tot_biomass"][:, ti] = np.tile(
                    np.mean(mangrove.bio_total_cell, axis=1), (1, len(y_dates))
                )[self.idx_stations, :]

            conditions_funct = dict(
                hydro_mor=update_hydro_mor,
                mangrove_characteristics=update_mangrove_characteristics,
            )

            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()
