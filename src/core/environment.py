"""
coral_mostoel - environment

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""

from pathlib import Path

import pandas as pd


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
