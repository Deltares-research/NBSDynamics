from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from netCDF4 import Dataset
from pandas import DataFrame

from src.core.base_model import BaseModel
from src.core.common.space_time import DataReshape
from src.core.vegetation.veg_model import Vegetation


class ModelParameters(BaseModel):
    lme: bool = False  # light micro-environment
    fme: bool = False  # flow micro-environment
    tme: bool = False  # thermal micro-environment
    pd: bool = False  # photosynthetic dependencies
    ps: bool = False  # population states
    calc: bool = False  # calcification rates
    md: bool = False  # morphological development



    def valid_output(self) -> bool:
        return any(self.dict().values())


class BaseOutput(BaseModel):
    """
    Base class containing the generic definition of a 'Vegetation' output model.
    """

    output_dir: Path
    output_filename: str

    # Output model attributes.
    output_params: ModelParameters = ModelParameters()

    def valid_output(self) -> bool:
        """
        Verifies whether this model can generate valid output.

        Returns:
            bool: Output is valid.
        """
        return self.output_params.valid_output()

    @property
    def output_filepath(self) -> Path:
        """
        Gets the full path to the output netcdf file.

        Returns:
            Path: Output .nc file.
        """
        return self.output_dir / self.output_filename


class MapOutput(BaseOutput):
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

    def initialize(self, veg_model: Vegetation):
        """Initiate mapping output file in which annual output covering the whole model domain is stored.

        """
        if not self.valid_output():
            return
        # Open netcdf data and initialize needed variables.
        with Dataset(self.output_filepath, "w", format="NETCDF4") as _map_data:
            _map_data.description = "Mapped simulation data of the VegetationModel."

            # dimensions
            _map_data.createDimension("time", None)
            _map_data.createDimension("nmesh2d_face", self.space)

            # variables
            t = _map_data.createVariable("time", int, ("time",))
            t.long_name = "year"
            t.units = "years since 0 B.C."

            x = _map_data.createVariable("nmesh2d_x", "f8", ("nmesh2d_face",))
            x.long_name = "x-coordinate"
            x.units = "m"

            y = _map_data.createVariable("nmesh2d_y", "f8", ("nmesh2d_face",))
            y.long_name = "y-coordinate"
            y.units = "m"

            t[:] = self.first_year
            x[:] = self.xy_coordinates[:, 0]
            y[:] = self.xy_coordinates[:, 1]

            # initial conditions
            # Definition of methods to initialize the netcdf variables.
            def init_cover():
                cover = _map_data.createVariable(
                    "cover", "f8", ("time", "nmesh2d_face")
                )
                cover.long_name = "sum of fraction of area coverage in each cell (for all ages)"
                cover.units = "-"
                cover[:, :] = 0 #could be =veg.cover if there is an initial one

            def init_height():
                sheight = _map_data.createVariable(
                    "stemheight", "f8", ("time", "nmesh2d_face")
                )
                sheight.long_name = "stem height"
                sheight.units = "m"
                sheight[:, :] = 0

            def init_diaveg():
                diaveg = _map_data.createVariable("diaveg", "f8", ("time", "nmesh2d_face"))
                diaveg.long_name = (
                    "stem diameter"
                )
                diaveg.units = "m"
                diaveg[:, :] = 0

            def init_rnveg():
                rnveg = _map_data.createVariable("rnveg", "f8", ("time", "nmesh2d_face"))
                rnveg.long_name = (
                    "plant density"
                )
                diaveg.units = "1/m2"
                diaveg[:, :] = 0

            def init_bl():
                bedlevel = _map_data.createVariable("bedlevel", "f8", ("time", "nmesh2d_face"))
                bedlevel.long_name = (
                    "plant density"
                )
                bedlevel.units = "1/m2"
                bedlevel[:, :] = 0

            conditions_funct = dict(
                cover=init_cover,
                sheight=init_height,
                diaveg=init_diaveg,
                rnveg=init_rnveg,
                bedlevel=init_bl

            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()

    def update(self, veg: Vegetation, period: int): ##TODO why not found?
        """Write data every period ran covering the whole model domain.

        :param veg: Vegetation
        :param period: based on veg_base_simulation

        :type veg: Vegetation
        :type period: int
        """
        if not self.valid_output():
            return
        with Dataset(self.output_filepath, mode="a") as _map_data:
            # i = int(year - self.first_year)
            i = int(period)
            _map_data["time"][i] = period

            def update_cover():
                _map_data["cover"][-1, :] = veg._cover[:, -1]

            def update_height():
                _map_data["sheight"][-1, :] = veg.veg_height

            def update_diaveg():
                _map_data["diaveg"][-1, :] = veg.veg_dia[:, -1]

            def update_rnveg():
                _map_data["rnveg"][-1, :] = veg.veg_dia[:, -1] ##TODO should it be stem_num?


            conditions_funct = dict(
                cover=update_cover,
                sheight=update_height,
                diaveg=update_diaveg,
                rnveg=update_rnveg,
                bedlevel=update_bl
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()


class HisOutput(BaseOutput):
    """
    Object representing a His output. Implements the 'OutputProtocol'.
    """

    output_filename = "CoralModel_his.nc"
    xy_stations: Optional[np.ndarray]
    idx_stations: Optional[np.ndarray]
    first_date: Optional[Union[np.datetime64, datetime]]

    def initialize(self, _: Vegetation):
        """Initiate history output file in which daily output at predefined locations within the model is stored."""
        if not self.valid_output():
            return
        with Dataset(self.output_filepath, "w", format="NETCDF4") as _his_data:
            _his_data.description = "Historic simulation data of the CoralModel"

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

            def init_cover():
                cover = _map_data.createVariable(
                    "cover", "f8", ("time", "nmesh2d_face")
                )
                cover.long_name = "sum of fraction of area coverage in each cell (for all ages)"
                cover.units = "-"
                cover[:, :] = 0  # could be =veg.cover if there is an initial one

            def init_height():
                sheight = _map_data.createVariable(
                    "stemheight", "f8", ("time", "nmesh2d_face")
                )
                sheight.long_name = "stem height"
                sheight.units = "m"
                sheight[:, :] = 0

            def init_diaveg():
                diaveg = _map_data.createVariable("diaveg", "f8", ("time", "nmesh2d_face"))
                diaveg.long_name = (
                    "stem diameter"
                )
                diaveg.units = "m"
                diaveg[:, :] = 0

            def init_rnveg():
                rnveg = _map_data.createVariable("rnveg", "f8", ("time", "nmesh2d_face"))
                rnveg.long_name = (
                    "plant density"
                )
                diaveg.units = "1/m2"
                diaveg[:, :] = 0

            def init_bl():
                bedlevel = _map_data.createVariable("bedlevel", "f8", ("time", "nmesh2d_face"))
                bedlevel.long_name = (
                    "plant density"
                )
                bedlevel.units = "1/m2"
                bedlevel[:, :] = 0

            #initial conditions
            conditions_funct = dict(
                cover=init_cover,
                sheight=init_height,
                diaveg=init_diaveg,
                rnveg=init_rnveg,
                bedlevel=init_bl

            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()

    def update(self, veg: Vegetation, dates: DataFrame): ##TODO leave as DataFrame? Finish the parameters
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
            ti = (y_dates - self.first_date).dt.days.values
            _his_data["time"][ti] = y_dates.values

            def update_cover():
                _his_data["Iz"][ti, :] = coral.light[self.idx_stations, :].transpose()

            def update_fme():
                _his_data["ucm"][ti, :] = np.tile(coral.ucm, (len(y_dates), 1))[
                    :, self.idx_stations
                ]

            def update_tme():
                _his_data["Tc"][ti, :] = coral.temp[self.idx_stations, :].transpose()
                if (
                    len(DataReshape.variable2array(coral.Tlo)) > 1
                    and len(DataReshape.variable2array(coral.Thi)) > 1
                ):
                    _his_data["Tlo"][ti, :] = np.tile(coral.Tlo, (len(y_dates), 1))[
                        :, self.idx_stations
                    ]
                    _his_data["Thi"][ti, :] = np.tile(coral.Thi, (len(y_dates), 1))[
                        :, self.idx_stations
                    ]
                else:
                    _his_data["Tlo"][ti, :] = coral.Tlo * np.ones(
                        (len(y_dates), len(self.idx_stations))
                    )
                    _his_data["Thi"][ti, :] = coral.Thi * np.ones(
                        (len(y_dates), len(self.idx_stations))
                    )

            def update_pd():
                _his_data["PD"][ti, :] = coral.photo_rate[
                    self.idx_stations, :
                ].transpose()

            def update_ps():
                _his_data["PT"][ti, :] = (
                    coral.pop_states[self.idx_stations, :, :].sum(axis=2).transpose()
                )
                _his_data["PH"][ti, :] = coral.pop_states[
                    self.idx_stations, :, 0
                ].transpose()
                _his_data["PR"][ti, :] = coral.pop_states[
                    self.idx_stations, :, 1
                ].transpose()
                _his_data["PP"][ti, :] = coral.pop_states[
                    self.idx_stations, :, 2
                ].transpose()
                _his_data["PB"][ti, :] = coral.pop_states[
                    self.idx_stations, :, 3
                ].transpose()



            conditions_funct = dict(
                cover=update_cover,
                sheight=update_height,
                diaveg=update_diaveg,
                rnveg=update_rnveg,
                bedlevel=update_bl
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()
