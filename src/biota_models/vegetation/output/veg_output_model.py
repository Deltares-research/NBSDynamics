from datetime import datetime
from typing import Optional, Union

import numpy as np
from netCDF4 import Dataset
from pandas import DataFrame

from src.biota_models.vegetation.model.veg_model import Vegetation
from src.core.output.base_output_model import BaseOutput, BaseOutputParameters


class VegetationOutputParameters(BaseOutputParameters):
    hydro_mor: bool = True  # hydro_morph micro-environment
    veg_characteristics: bool = True  # vegetation characteristics


class _VegetationOutput(BaseOutput):
    """
    Base class containing the generic definition of a 'Vegetation' output model.
    """

    # Output model attributes.
    output_params: VegetationOutputParameters = VegetationOutputParameters()


class VegetationMapOutput(_VegetationOutput):
    """
    Object representing a Map output. Implements the 'OutputProtocol'.
    """

    output_filename = "VegModel_map.nc"
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

    def initialize(self, vegetation: Optional[Vegetation]):
        self.output_filename = "VegModel_"+ vegetation.species +"_map.nc"
        """Initiate mapping output file in which output covering the whole model domain is stored every period of running."""
        if not self.valid_output():
            return
        # Open netcdf data and initialize needed variables.
        with Dataset(self.output_filepath, "w", format="NETCDF4") as _map_data:
            _map_data.description = "Mapped simulation data of the VegetationModel."

            # dimensions
            _map_data.createDimension("time", None)
            _map_data.createDimension("nmesh2d_face", self.space)
            _map_data.createDimension("age", None)

            # variables
            t = _map_data.createVariable("time", int, ("time",))
            t.long_name = "period"
            t.units = "period"

            age = _map_data.createVariable("age", int, ("age"))
            age.long_name = "vegetation age"
            age.units = "days"

            x = _map_data.createVariable("nmesh2d_x", "f8", ("nmesh2d_face",))
            x.long_name = "x-coordinate"
            x.units = "m"

            y = _map_data.createVariable("nmesh2d_y", "f8", ("nmesh2d_face",))
            y.long_name = "y-coordinate"
            y.units = "m"

            # t[:] = self.first_year
            x[:] = self.xy_coordinates[:, 0]
            y[:] = self.xy_coordinates[:, 1]

            # initial conditions
            # Definition of methods to initialize the netcdf variables.

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

            def init_veg_characteristics():
                cover = _map_data.createVariable(
                    "cover", "f8", ("time", "nmesh2d_face")
                )
                cover.long_name = "sum of fraction coverage in each cell (for all ages)"
                cover.units = "-"
                cover[:, :] = 0  # could be =veg.cover if there is an initial one

                height = _map_data.createVariable(
                    "height", "f8", ("time", "nmesh2d_face")
                )
                height.long_name = "vegetation height"
                height.units = "m"
                height[:, :] = 0

                diaveg = _map_data.createVariable(
                    "diaveg", "f8", ("time", "nmesh2d_face")
                )
                diaveg.long_name = "stem diameter"
                diaveg.units = "m"
                diaveg[:, :] = 0

                rnveg = _map_data.createVariable(
                    "rnveg", "f8", ("time", "nmesh2d_face")
                )
                rnveg.long_name = "vegetation density"
                diaveg.units = "1/m2"
                diaveg[:, :] = 0

                veg_frac_j = _map_data.createVariable(
                    "veg_frac_j", "f8", ("nmesh2d_face", "age", "time")
                )
                veg_frac_j.long_name = (
                    "Vegetation fraction in each growth day for juvenile"
                )
                veg_frac_j.units = "-"
                veg_frac_j[:, :, :] = 0
                veg_frac_m = _map_data.createVariable(
                    "veg_frac_m", "f8", ("nmesh2d_face", "age", "time")
                )
                veg_frac_m.long_name = (
                    "Vegetation fraction in each growth day for mature"
                )
                veg_frac_m.units = "-"
                veg_frac_m[:, :, :] = 0

            conditions_funct = dict(
                hydro_mor=init_hydro_mor,
                veg_characteristics=init_veg_characteristics,
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()

    def update(self, veg: Vegetation, end_time: int, ets, year, constants):
        """Write data every period ran covering the whole model domain.

        :param veg: Vegetation
        :param period: based on veg_base_simulation

        :type veg: Vegetation
        :type period: int
        """
        if not self.valid_output():
            return
        with Dataset(self.output_filepath, mode="a") as _map_data:
            i = ets + constants.t_eco_year * year

            _map_data["time"][i] = end_time

            def update_hydro_mor():
                _map_data["max_tau"][-1, :] = veg.max_tau
                _map_data["max_u"][-1, :] = veg.max_u
                _map_data["max_wl"][-1, :] = veg.max_wl
                _map_data["min_wl"][-1, :] = veg.min_wl
                _map_data["bl"][-1, :] = veg.bl

            def update_veg_characteristics():
                _map_data["cover"][-1, :] = veg.total_cover.transpose()
                _map_data["height"][-1, :] = veg.av_height.transpose()
                _map_data["diaveg"][-1, :] = veg.av_stemdia.transpose()
                _map_data["rnveg"][-1, :] = veg.veg_den.transpose()
                _map_data["veg_frac_j"][:, :, -1] = veg.juvenile.veg_frac[:, :]
                _map_data["veg_frac_m"][:, :, -1] = veg.mature.veg_frac[:, :]

            conditions_funct = dict(
                hydro_mor=update_hydro_mor,
                veg_characteristics=update_veg_characteristics,
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()


class VegetationHisOutput(_VegetationOutput):
    """
    Object representing a His output. Implements the 'OutputProtocol'.
    """

    output_filename = "VegModel_his.nc"
    xy_stations: Optional[np.ndarray]
    idx_stations: Optional[np.ndarray]
    first_date: Optional[Union[np.datetime64, datetime]]

    def initialize(self, veg: Vegetation):
        """Initiate history output file in which daily output at predefined locations within the model is stored."""
        self.output_filename = "VegModel_" + veg.species + "_his.nc"
        if not self.valid_output():
            return
        with Dataset(self.output_filepath, "w", format="NETCDF4") as _his_data:
            _his_data.description = "Historic simulation data of the VegetaionModel"

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
                max_tau = _his_data.createVariable(
                    "max_tau", "f8", ("time", "stations")
                )
                max_tau.long_name = "maximum bed shear stress"
                max_tau.units = "N/m^2"
                max_tau[:, :] = 0
                max_u = _his_data.createVariable("max_u", "f8", ("time", "stations"))
                max_u.long_name = "maximum flow velocity"
                max_u.units = "m/s"
                max_u[:, :] = 0
                max_wl = _his_data.createVariable("max_wl", "f8", ("time", "stations"))
                max_wl.long_name = "maximum water level"
                max_wl.units = "m"
                max_wl[:, :] = 0
                min_wl = _his_data.createVariable("min_wl", "f8", ("time", "stations"))
                min_wl.long_name = "minimum water level"
                min_wl.units = "m"
                min_wl[:, :] = 0
                bl = _his_data.createVariable("bl", "f8", ("time", "stations"))
                bl.long_name = "bedlevel"
                bl.units = "m"
                bl[:, :] = 0

            def init_veg_characteristics():
                cover = _his_data.createVariable("cover", "f8", ("time", "stations"))
                cover.long_name = "sum of fraction coverage in each cell (for all ages)"
                cover.units = "-"
                cover[
                    :, :
                ] = veg.total_cover  # could be =veg.cover if there is an initial one

                height = _his_data.createVariable("height", "f8", ("time", "stations"))
                height.long_name = "vegetation height"
                height.units = "m"
                height[:, :] = 0

                diaveg = _his_data.createVariable("diaveg", "f8", ("time", "stations"))
                diaveg.long_name = "stem diameter"
                diaveg.units = "m"
                diaveg[:, :] = 0

                rnveg = _his_data.createVariable("rnveg", "f8", ("time", "stations"))
                rnveg.long_name = "vegetation density"
                diaveg.units = "1/m2"
                diaveg[:, :] = 0

                veg_frac_j = _his_data.createVariable(
                    "veg_frac_j", "f8", ("time", "stations")
                )
                veg_frac_j.long_name = (
                    "Vegetation fraction in each growth day for juvenile"
                )
                veg_frac_j.units = "-"
                veg_frac_j[:, :] = 0
                veg_frac_m = _his_data.createVariable(
                    "veg_frac_m", "f8", ("time", "stations")
                )
                veg_frac_m.long_name = (
                    "Vegetation fraction in each growth day for mature"
                )
                veg_frac_m.units = "-"
                veg_frac_m[:, :] = 0

            conditions_funct = dict(
                hydro_mor=init_hydro_mor,
                veg_characteristics=init_veg_characteristics,
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()

    def update(self, veg: Vegetation, dates: DataFrame):
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
                _his_data["max_tau"][ti, :] = np.tile(veg.max_tau, (len(y_dates), 1))[
                    :, self.idx_stations
                ]
                _his_data["max_u"][ti, :] = np.tile(veg.max_u, (len(y_dates), 1))[
                    :, self.idx_stations
                ]
                _his_data["max_wl"][ti, :] = np.tile(veg.max_wl, (len(y_dates), 1))[
                    :, self.idx_stations
                ]
                _his_data["min_wl"][ti, :] = np.tile(veg.min_wl, (len(y_dates), 1))[
                    :, self.idx_stations
                ]
                _his_data["bl"][ti, :] = np.tile(veg.bl, (len(y_dates), 1))[
                    :, self.idx_stations
                ]

            def update_veg_characteristics():
                _his_data["cover"][ti, :] = np.tile(
                    veg.total_cover.transpose(), (len(y_dates), 1)
                )[:, self.idx_stations]
                _his_data["height"][ti, :] = np.tile(
                    veg.av_height.transpose(), (len(y_dates), 1)
                )[:, self.idx_stations]
                _his_data["diaveg"][ti, :] = np.tile(
                    veg.av_stemdia.transpose(), (len(y_dates), 1)
                )[:, self.idx_stations]
                _his_data["rnveg"][ti, :] = np.tile(
                    veg.veg_den.transpose(), (len(y_dates), 1)
                )[:, self.idx_stations]
                _his_data["veg_frac_j"][ti, :] = np.tile(
                    veg.juvenile.cover.transpose(), (len(y_dates), 1)
                )[:, self.idx_stations]
                _his_data["veg_frac_m"][ti, :] = np.tile(
                    veg.mature.cover.transpose(), (len(y_dates), 1)
                )[:, self.idx_stations]

            conditions_funct = dict(
                hydro_mor=update_hydro_mor,
                veg_characteristics=update_veg_characteristics,
            )

            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()
