from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from netCDF4 import Dataset
from pandas import DataFrame

from src.core.base_model import BaseModel
from src.core.biota.coral.coral_model import Coral
from src.core.common.space_time import DataReshape


class ModelParameters(BaseModel):
    lme: bool = True  # light micro-environment
    fme: bool = True  # flow micro-environment
    tme: bool = True  # thermal micro-environment
    pd: bool = True  # photosynthetic dependencies
    ps: bool = True  # population states
    calc: bool = True  # calcification rates
    md: bool = True  # morphological development

    def valid_output(self) -> bool:
        return any(self.dict().values())


class BaseOutput(BaseModel):
    """
    Base class containing the generic definition of a 'Coral' output model.
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

    output_filename = "CoralModel_map.nc"
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

    def initialize(self, coral: Coral):
        """Initiate mapping output file in which annual output covering the whole model domain is stored.

        :param coral: coral animal
        :type coral: Coral
        """
        if not self.valid_output():
            return
        # Open netcdf data and initialize needed variables.
        with Dataset(self.output_filepath, "w", format="NETCDF4") as _map_data:
            _map_data.description = "Mapped simulation data of the CoralModel."

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
            def init_lme():
                light_set = _map_data.createVariable(
                    "Iz", "f8", ("time", "nmesh2d_face")
                )
                light_set.long_name = "annual mean representative light-intensity"
                light_set.units = "micro-mol photons m-2 s-1"
                light_set[:, :] = 0

            def init_fme():
                flow_set = _map_data.createVariable(
                    "ucm", "f8", ("time", "nmesh2d_face")
                )
                flow_set.long_name = "annual mean in-canopy flow"
                flow_set.units = "m s-1"
                flow_set[:, :] = 0

            def init_tme():
                temp_set = _map_data.createVariable(
                    "Tc", "f8", ("time", "nmesh2d_face")
                )
                temp_set.long_name = "annual mean coral temperature"
                temp_set.units = "K"
                temp_set[:, :] = 0

                low_temp_set = _map_data.createVariable(
                    "Tlo", "f8", ("time", "nmesh2d_face")
                )
                low_temp_set.long_name = "annual mean lower thermal limit"
                low_temp_set.units = "K"
                low_temp_set[:, :] = 0

                high_temp_set = _map_data.createVariable(
                    "Thi", "f8", ("time", "nmesh2d_face")
                )
                high_temp_set.long_name = "annual mean upper thermal limit"
                high_temp_set.units = "K"
                high_temp_set[:, :] = 0

            def init_pd():
                pd_set = _map_data.createVariable("PD", "f8", ("time", "nmesh2d_face"))
                pd_set.long_name = "annual sum photosynthetic rate"
                pd_set.units = "-"
                pd_set[:, :] = 0

            def init_ps():
                pt_set = _map_data.createVariable("PT", "f8", ("time", "nmesh2d_face"))
                pt_set.long_name = (
                    "total living coral population at the end of the year"
                )
                pt_set.units = "-"
                pt_set[:, :] = coral.living_cover

                ph_set = _map_data.createVariable("PH", "f8", ("time", "nmesh2d_face"))
                ph_set.long_name = "healthy coral population at the end of the year"
                ph_set.units = "-"
                ph_set[:, :] = coral.living_cover

                pr_set = _map_data.createVariable("PR", "f8", ("time", "nmesh2d_face"))
                pr_set.long_name = "recovering coral population at the end of the year"
                pr_set.units = "-"
                pr_set[:, :] = 0

                pp_set = _map_data.createVariable("PP", "f8", ("time", "nmesh2d_face"))
                pp_set.long_name = "pale coral population at the end of the year"
                pp_set.units = "-"
                pp_set[:, :] = 0

                pb_set = _map_data.createVariable("PB", "f8", ("time", "nmesh2d_face"))
                pb_set.long_name = "bleached coral population at the end of the year"
                pb_set.units = "-"
                pb_set[:, :] = 0

            def init_calc():
                calc_set = _map_data.createVariable(
                    "calc", "f8", ("time", "nmesh2d_face")
                )
                calc_set.long_name = "annual sum calcification rate"
                calc_set.units = "kg m-2 yr-1"
                calc_set[:, :] = 0

            def init_md():
                dc_set = _map_data.createVariable("dc", "f8", ("time", "nmesh2d_face"))
                dc_set.long_name = "coral plate diameter"
                dc_set.units = "m"
                dc_set[0, :] = coral.dc

                hc_set = _map_data.createVariable("hc", "f8", ("time", "nmesh2d_face"))
                hc_set.long_name = "coral height"
                hc_set.units = "m"
                hc_set[0, :] = coral.hc

                bc_set = _map_data.createVariable("bc", "f8", ("time", "nmesh2d_face"))
                bc_set.long_name = "coral base diameter"
                bc_set.units = "m"
                bc_set[0, :] = coral.bc

                tc_set = _map_data.createVariable("tc", "f8", ("time", "nmesh2d_face"))
                tc_set.long_name = "coral plate thickness"
                tc_set.units = "m"
                tc_set[0, :] = coral.tc

                ac_set = _map_data.createVariable("ac", "f8", ("time", "nmesh2d_face"))
                ac_set.long_name = "coral axial distance"
                ac_set.units = "m"
                ac_set[0, :] = coral.ac

                vc_set = _map_data.createVariable("Vc", "f8", ("time", "nmesh2d_face"))
                vc_set.long_name = "coral volume"
                vc_set.units = "m3"
                vc_set[0, :] = coral.volume

            conditions_funct = dict(
                lme=init_lme,
                fme=init_fme,
                tme=init_tme,
                pd=init_pd,
                ps=init_ps,
                calc=init_calc,
                md=init_md,
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()

    def update(self, coral: Coral, year: int):
        """Write data as annual output covering the whole model domain.

        :param coral: coral animal
        :param year: simulation year

        :type coral: Coral
        :type year: int
        """
        if not self.valid_output():
            return
        with Dataset(self.output_filepath, mode="a") as _map_data:
            i = int(year - self.first_year)
            _map_data["time"][i] = year

            def update_lme():
                _map_data["Iz"][-1, :] = coral.light[:, -1]

            def update_fme():
                _map_data["ucm"][-1, :] = coral.ucm

            def update_tme():
                _map_data["Tc"][-1, :] = coral.temp[:, -1]
                _map_data["Tlo"][-1, :] = (
                    coral.Tlo
                    if len(DataReshape.variable2array(coral.Tlo)) > 1
                    else coral.Tlo * np.ones(self.space)
                )
                _map_data["Thi"][-1, :] = (
                    coral.Thi
                    if len(DataReshape.variable2array(coral.Thi)) > 1
                    else coral.Thi * np.ones(self.space)
                )

            def update_pd():
                _map_data["PD"][-1, :] = coral.photo_rate.mean(axis=1)

            def update_ps():
                _map_data["PT"][-1, :] = coral.pop_states[:, -1, :].sum(axis=1)
                _map_data["PH"][-1, :] = coral.pop_states[:, -1, 0]
                _map_data["PR"][-1, :] = coral.pop_states[:, -1, 1]
                _map_data["PP"][-1, :] = coral.pop_states[:, -1, 2]
                _map_data["PB"][-1, :] = coral.pop_states[:, -1, 3]

            def update_calc():
                _map_data["calc"][-1, :] = coral.calc.sum(axis=1)

            def update_md():
                _map_data["dc"][-1, :] = coral.dc
                _map_data["hc"][-1, :] = coral.hc
                _map_data["bc"][-1, :] = coral.bc
                _map_data["tc"][-1, :] = coral.tc
                _map_data["ac"][-1, :] = coral.ac
                _map_data["Vc"][-1, :] = coral.volume

            conditions_funct = dict(
                lme=update_lme,
                fme=update_fme,
                tme=update_tme,
                pd=update_pd,
                ps=update_ps,
                calc=update_calc,
                md=update_md,
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

    def initialize(self, _: Coral):
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

            def init_lme():
                light_set = _his_data.createVariable("Iz", "f8", ("time", "stations"))
                light_set.long_name = "representative light-intensity"
                light_set.units = "micro-mol photons m-2 s-1"

            def init_fme():
                flow_set = _his_data.createVariable("ucm", "f8", ("time", "stations"))
                flow_set.long_name = "in-canopy flow"
                flow_set.units = "m s-1"

            def init_tme():
                temp_set = _his_data.createVariable("Tc", "f8", ("time", "stations"))
                temp_set.long_name = "coral temperature"
                temp_set.units = "K"

                low_temp_set = _his_data.createVariable(
                    "Tlo", "f8", ("time", "stations")
                )
                low_temp_set.long_name = "lower thermal limit"
                low_temp_set.units = "K"

                high_temp_set = _his_data.createVariable(
                    "Thi", "f8", ("time", "stations")
                )
                high_temp_set.long_name = "upper thermal limit"
                high_temp_set.units = "K"

            def init_pd():
                pd_set = _his_data.createVariable("PD", "f8", ("time", "stations"))
                pd_set.long_name = "photosynthetic rate"
                pd_set.units = "-"

            def init_ps():
                pt_set = _his_data.createVariable("PT", "f8", ("time", "stations"))
                pt_set.long_name = "total coral population"
                pt_set.units = "-"

                ph_set = _his_data.createVariable("PH", "f8", ("time", "stations"))
                ph_set.long_name = "healthy coral population"
                ph_set.units = "-"

                pr_set = _his_data.createVariable("PR", "f8", ("time", "stations"))
                pr_set.long_name = "recovering coral population"
                pr_set.units = "-"

                pp_set = _his_data.createVariable("PP", "f8", ("time", "stations"))
                pp_set.long_name = "pale coral population"
                pp_set.units = "-"

                pb_set = _his_data.createVariable("PB", "f8", ("time", "stations"))
                pb_set.long_name = "bleached coral population"
                pb_set.units = "-"

            def init_calc():
                calc_set = _his_data.createVariable("G", "f8", ("time", "stations"))
                calc_set.long_name = "calcification"
                calc_set.units = "kg m-2 d-1"

            def init_md():
                dc_set = _his_data.createVariable("dc", "f8", ("time", "stations"))
                dc_set.long_name = "coral plate diameter"
                dc_set.units = "m"

                hc_set = _his_data.createVariable("hc", "f8", ("time", "stations"))
                hc_set.long_name = "coral height"
                hc_set.units = "m"

                bc_set = _his_data.createVariable("bc", "f8", ("time", "stations"))
                bc_set.long_name = "coral base diameter"
                bc_set.units = "m"

                tc_set = _his_data.createVariable("tc", "f8", ("time", "stations"))
                tc_set.long_name = "coral plate thickness"
                tc_set.units = "m"

                ac_set = _his_data.createVariable("ac", "f8", ("time", "stations"))
                ac_set.long_name = "coral axial distance"
                ac_set.units = "m"

                vc_set = _his_data.createVariable("Vc", "f8", ("time", "stations"))
                vc_set.long_name = "coral volume"
                vc_set.units = "m3"

            # initial conditions
            conditions_funct = dict(
                lme=init_lme,
                fme=init_fme,
                tme=init_tme,
                pd=init_pd,
                ps=init_ps,
                calc=init_calc,
                md=init_md,
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()

    def update(self, coral: Coral, dates: DataFrame):
        """Write data as daily output at predefined locations within the model domain.

        :param coral: coral animal
        :param dates: dates of simulation year

        :type coral: Coral
        :type dates: DataFrame
        """
        if not self.valid_output():
            return
        with Dataset(self.output_filepath, mode="a") as _his_data:
            y_dates = dates.reset_index(drop=True)
            ti = (y_dates - self.first_date).dt.days.values
            _his_data["time"][ti] = y_dates.values

            def update_lme():
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

            def update_calc():
                _his_data["G"][ti, :] = coral.calc[self.idx_stations, :].transpose()

            def update_md():
                _his_data["dc"][ti, :] = np.tile(coral.dc, (len(y_dates), 1))[
                    :, self.idx_stations
                ]
                _his_data["hc"][ti, :] = np.tile(coral.hc, (len(y_dates), 1))[
                    :, self.idx_stations
                ]
                _his_data["bc"][ti, :] = np.tile(coral.bc, (len(y_dates), 1))[
                    :, self.idx_stations
                ]
                _his_data["tc"][ti, :] = np.tile(coral.tc, (len(y_dates), 1))[
                    :, self.idx_stations
                ]
                _his_data["ac"][ti, :] = np.tile(coral.ac, (len(y_dates), 1))[
                    :, self.idx_stations
                ]
                _his_data["Vc"][ti, :] = np.tile(coral.volume, (len(y_dates), 1))[
                    :, self.idx_stations
                ]

            conditions_funct = dict(
                lme=update_lme,
                fme=update_fme,
                tme=update_tme,
                pd=update_pd,
                ps=update_ps,
                calc=update_calc,
                md=update_md,
            )
            for key, v_func in conditions_funct.items():
                if self.output_params.dict()[key]:
                    v_func()
