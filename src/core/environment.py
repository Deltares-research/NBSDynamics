"""
coral_mostoel - environment

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""

import distutils.util as du
from pathlib import Path
from src.core.base_model import BaseModel
from pydantic import root_validator
from typing import Optional
import numpy as np
import pandas as pd


class Constants(BaseModel):
    """Object containing all constants used in coral_model simulations."""

    # Input file
    input_file: Optional[Path]

    # Processes
    fme: bool = False
    tme: bool = False
    pfd: bool = False
    warn_proc: bool = True

    # light micro-environment
    Kd0: float = 0.1
    theta_max: float = 0.5 * np.pi

    # flow micro-environment
    Cs: float = 0.17
    Cm: float = 1.7
    Cf: float = 0.01
    nu: float = 1e-6
    alpha: float = 1e-7
    psi: float = 2
    wcAngle: float = 0.0
    rd: float = 500
    numericTheta: float = 0.5
    err: float = 1e-3
    maxiter_k: np.int32 = 1e5
    maxiter_aw: np.int32 = 1e5

    # thermal micro-environment
    K0: float = 80.0
    ap: float = 0.4
    k: float = 0.6089

    # photosynthetic light dependency
    iota: float = 0.6
    ik_max: float = 372.32
    pm_max: float = 1.0
    betaI: float = 0.34
    betaP: float = 0.09
    Icomp: float = 0.01

    # photosynthetic thermal dependency
    Ea: float = 6e4
    R: float = 8.31446261815324
    k_var: float = 2.45
    nn: float = 60

    # photosynthetic flow dependency
    pfd_min: float = 0.68886964
    ucr: float = 0.5173

    # population dynamics
    r_growth: float = 0.002
    r_recovery: float = 0.2
    r_mortality: float = 0.04
    r_bleaching: float = 8.0

    # calcification
    gC: float = 0.5
    omegaA0: float = 5.0
    omega0: float = 0.14587415
    kappaA: float = 0.66236107

    # morphological development
    rf: float = 1.0
    rp: float = 1.0
    prop_form: float = 0.1
    prop_plate: float = 0.5
    prop_plate_flow: float = 0.1
    prop_space: float = 0.5 / np.sqrt(2.0)
    prop_space_light: float = 0.1
    prop_space_flow: float = 0.1
    u0: float = 0.2
    rho_c: float = 1600.0

    # dislodgement criterion
    sigma_t: float = 2e5
    Cd: float = 1.0
    rho_w: float = 1025.0

    # coral recruitment
    no_larvae: float = 1e6
    prob_settle: float = 1e-4
    d_larvae: float = 1e-3

    @root_validator
    @classmethod
    def check_processes(cls, values: dict) -> dict:
        """
        Validates the input values so that the processes are compatible between themselves.

        Args:
            values (dict): Dictionary of values already validated individually.

        Returns:
            dict: Dictionary of validated values as a whole.
        """
        if not values["pfd"]:
            if values["fme"] and values["warn_proc"]:
                print(
                    "WARNING: Flow micro-environment (FME) not possible "
                    "when photosynthetic flow dependency (PFD) is disabled."
                )
            values["fme"] = False
            values["tme"] = False

        else:
            if not values["fme"]:
                if values["tme"] and values["warn_proc"]:
                    print(
                        "WARNING: Thermal micro-environment (TME) not possible "
                        "when flow micro-environment is disabled."
                    )
                values["tme"] = False

        if values["tme"] and values["warn_proc"]:
            print("WARNING: Thermal micro-environment not fully implemented yet.")

        if not values["pfd"] and values["warn_proc"]:
            print(
                "WARNING: Exclusion of photosynthetic flow dependency not fully implemented yet."
            )
        return values

    @classmethod
    def from_input_file(cls, input_file: Path):
        """
        Generates a 'Constants' class based on the defined parameters in the input_file.

        Args:
            input_file (Path): Path to the constants input (.txt) file.
        """

        def split_line(line: str):
            s_line = line.split("=")
            if len(s_line) <= 1:
                raise ValueError
            return s_line[0].strip(), s_line[1].strip()

        def format_line(line: str) -> str:
            return split_line(line.split("#")[0])

        def normalize_line(line: str) -> str:
            return line.strip()

        input_lines = [
            format_line(n_line)
            for line in input_file.read_text().splitlines(keepends=False)
            if line and not (n_line := normalize_line(line)).startswith("#")
        ]
        return cls(**dict(input_lines))


class Environment:
    # TODO: Make this class robust

    _dates = None
    _light = None
    _light_attenuation = None
    _temperature = None
    _aragonite = None
    _storm_category = None

    @property
    def light(self):
        """Light-intensity in micro-mol photons per square metre-second."""
        return self._light

    @property
    def light_attenuation(self):
        """Light-attenuation coefficient in per metre."""
        return self._light_attenuation

    @property
    def temperature(self):
        """Temperature time-series in either Celsius or Kelvin."""
        return self._temperature

    @property
    def aragonite(self):
        """Aragonite saturation state."""
        return self._aragonite

    @property
    def storm_category(self):
        """Storm category time-series."""
        return self._storm_category

    @property
    def temp_kelvin(self):
        """Temperature in Kelvin."""
        if all(self.temperature.values < 100) and self.temperature is not None:
            return self.temperature + 273.15
        return self.temperature

    @property
    def temp_celsius(self):
        """Temperature in Celsius."""
        if all(self.temperature.values > 100) and self.temperature is not None:
            return self.temperature - 273.15
        return self.temperature

    @property
    def temp_mmm(self):
        monthly_mean = self.temp_kelvin.groupby(
            [self.temp_kelvin.index.year, self.temp_kelvin.index.month]
        ).agg(["mean"])
        monthly_maximum_mean = monthly_mean.groupby(level=0).agg(["min", "max"])
        monthly_maximum_mean.columns = monthly_maximum_mean.columns.droplevel([0, 1])
        return monthly_maximum_mean

    @property
    def dates(self):
        """Dates of time-series."""
        if self._dates is not None:
            d = self._dates
        elif self.light is not None:
            # TODO: Check column name of light-file
            d = self.light.reset_index().drop("light", axis=1)
            self._dates = d
        elif self.temperature is not None:
            d = self.temperature.reset_index().drop("sst", axis=1)
            self._dates = d
        else:
            msg = f"No initial data on dates provided."
            raise ValueError(msg)
        return pd.to_datetime(d["date"])

    def set_dates(self, start_date, end_date):
        """Set dates manually, ignoring possible dates in environmental time-series.

        :param start_date: first date of time-series
        :param end_date: last date of time-series

        :type start_date: str, datetime.date
        :type end_date: str, datetime.date
        """
        dates = pd.date_range(start_date, end_date, freq="D")
        self._dates = pd.DataFrame({"date": dates})

    def set_parameter_values(self, parameter, value, pre_date=None):
        """Set the time-series data to a time-series, or a  value. In case :param value: is not iterable, the
        :param parameter: is assumed to be constant over time. In case :param value: is iterable, make sure its length
        complies with the simulation length.

        Included parameters:
            light                       :   incoming light-intensity [umol photons m-2 s-1]
            LAC / light_attenuation     :   light attenuation coefficient [m-1]
            temperature                 :   sea surface temperature [K]
            aragonite                   :   aragonite saturation state [-]
            storm                       :   storm category, annually [-]

        :param parameter: parameter to be set
        :param value:  value
        :param pre_date: time-series start before simulation dates [yrs]

        :type parameter: str
        :type value: float, list, tuple, numpy.ndarray, pandas.DataFrame
        :type pre_date: None, int, optional
        """

        def set_value(val):
            """Function to set  value."""
            if pre_date is None:
                return pd.DataFrame({parameter: val}, index=self.dates)

            dates = pd.date_range(
                self.dates.iloc[0] - pd.DateOffset(years=pre_date),
                self.dates.iloc[-1],
                freq="D",
            )
            return pd.DataFrame({parameter: val}, index=dates)

        if self._dates is None:
            msg = (
                f"No dates are defined. "
                f"Please, first specify the dates before setting the time-series of {parameter}; "
                f'or make use of the "from_file"-method.'
            )
            raise TypeError(msg)

        if parameter == "LAC":
            parameter = "light_attenuation"

        daily_params = ("light", "light_attenuation", "temperature", "aragonite")
        if parameter in daily_params:
            setattr(self, f"_{parameter}", set_value(value))
        elif parameter == "storm":
            years = set(self.dates.dt.year)
            self._storm_category = pd.DataFrame(data=value, index=years)
        else:
            msg = f"Entered parameter ({parameter}) not included. See documentation."
            raise ValueError(msg)

    def from_file(self, parameter: str, file: Path):
        """Read the time-series data from a file.

        Included parameters:
            light                       :   incoming light-intensity [umol photons m-2 s-1]
            LAC / light_attenuation     :   light attenuation coefficient [m-1]
            temperature                 :   sea surface temperature [K]
            aragonite                   :   aragonite saturation state [-]
            storm                       :   storm category, annually [-]

        :param parameter: parameter to be read from file
        :param file: file name, incl. file extension

        :type parameter: str
        :type file: str
        """
        # TODO: Include functionality to check file's existence
        #  > certain files are necessary: light, temperature

        if not file.exists():
            raise FileNotFoundError(file)

        def read_index(fil):
            """Function applicable to time-series in Pandas."""
            time_series = pd.read_csv(fil, sep="\t")
            if time_series.isnull().values.any():
                msg = f"NaNs detected in time series {fil}"
                raise ValueError(msg)
            time_series["date"] = pd.to_datetime(time_series["date"])
            time_series.set_index("date", inplace=True)
            return time_series

        if parameter == "LAC":
            parameter = "light_attenuation"

        daily_params = ("light", "light_attenuation", "temperature", "aragonite")
        if parameter in daily_params:
            setattr(self, f"_{parameter}", read_index(file))
        elif parameter == "storm":
            self._storm_category = pd.read_csv(file, sep="\t")
            self._storm_category.set_index("year", inplace=True)
        else:
            msg = f"Entered parameter ({parameter}) not included. See documentation."
            raise ValueError(msg)
