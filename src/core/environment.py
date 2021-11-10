"""
coral_mostoel - environment

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
from src.core.base_model import BaseModel
from typing import Any, Iterable, Optional, Union
from pydantic import validator

from src.core.base_model import BaseModel

EnvInputAttr = Union[pd.DataFrame, Path]


class Environment(BaseModel):
    dates: Optional[pd.DataFrame] = ("1990, 01, 01", "2021, 12, 20")
    light: Optional[pd.DataFrame]
    light_attenuation: Optional[pd.DataFrame]
    temperature: Optional[pd.DataFrame]
    aragonite: Optional[pd.DataFrame]
    storm_category: Optional[pd.DataFrame]

    @validator("light", "light_attenuation", "temperature", "aragonite", pre=True)
    @classmethod
    def validate_dataframe_or_path(cls, value: EnvInputAttr) -> pd.DataFrame:
        """
        Transforms an input into the expected type for the parameter. In case a file it's provided
        it's content is converted into a pandas DataFrame.

        Args:
            value (EnvInputAttr): Value to be validated (Union[pd.DataFrame, Path]).

        Raises:
            FileNotFoundError: When the provided value is a non-existent Path.
            NotImplementedError: When the provided value is not supported.

        Returns:
            pd.DataFrame: Validated attribute value.
        """

        def read_index(value_file: Path) -> pd.DataFrame:
            """Function applicable to time-series in Pandas."""
            time_series = pd.read_csv(value_file, sep="\t")
            if time_series.isnull().values.any():
                msg = f"NaNs detected in time series {value_file}"
                raise ValueError(msg)
            time_series["date"] = pd.to_datetime(time_series["date"])
            time_series.set_index("date", inplace=True)
            return time_series

        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, Path):
            if not value.is_file():
                raise FileNotFoundError(value)
            return read_index(value)
        raise NotImplementedError(f"Validator not available for type {type(value)}")

    @validator("storm_category", pre=True)
    @classmethod
    def validate_storm_category(cls, value: EnvInputAttr) -> pd.DataFrame:
        """
        Transforms the input value given for the 'storm_category' parameter
        into a valid 'Environment' attribute.

        Args:
            value (EnvInputAttr): Value assigned to the attribute (Union[pd.DataFrame, Path]).

        Raises:
            FileNotFoundError: When the provided value is a non-existent Path.
            NotImplementedError: When the provided value is not supported.

        Returns:
            pd.DataFrame: Validated value.
        """
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, Path):
            if not value.is_file():
                raise FileNotFoundError(value)
            csv_values = pd.read_csv(value, sep="\t")
            csv_values.set_index("year", inplace=True)
            return csv_values
        raise NotImplementedError(f"Validator not available for type {type(value)}")

    @validator("dates", pre=True)
    @classmethod
    def prevalidate_dates(
        cls, value: Union[pd.DataFrame, Iterable[Union[str, datetime]]]
    ) -> pd.DataFrame:
        """
        Prevalidates the the input value given for the 'dates' parameter transforming it
        into a valid 'Environment' attribute.

        Args:
            value (Union[pd.DataFrame, Iterable[Union[str, datetime]]]): Value assigned to the attribute.

        Raises:
            NotImplementedError: When the provided value is not supported.

        Returns:
            pd.DataFrame: Validated value.
        """
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, Iterable):
            return cls.get_dates_dataframe(value[0], value[-1])
        raise NotImplementedError(f"Validator not available for type {type(value)}")

    @validator("dates", always=True, pre=False)
    @classmethod
    def check_dates(cls, v: Optional[pd.DataFrame], values: dict) -> pd.DataFrame:
        """
        Validates the dates value (post-process).

        Args:
            v (Optional[pd.DataFrame]): Value pre-validated for dates (if any).
            values (dict): Dictionary containing the rest of values given to initialize 'Environment'.

        Returns:
            pd.DataFrame: Validated dates value.
        """
        # Validate dates have values.
        if isinstance(v, pd.DataFrame):
            return v
        light_value: pd.DataFrame = values.get("light", None)
        if light_value is not None:
            # TODO: Check column name of light-file
            return light_value.reset_index().drop("light", axis=1)

        temp_value: pd.DataFrame = values.get("temperature", None)
        if temp_value is not None:
            return temp_value.reset_index().drop("sst", axis=1)

        return None

    @staticmethod
    def get_dates_dataframe(
        start_date: Union[str, datetime], end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        dates = pd.date_range(start_date, end_date, freq="D")
        return pd.DataFrame({"date": dates})

    def get_dates(self) -> Iterable[datetime]:
        """
        Just a shortcut being used in some occasions to get the datetime series array.

        Raises:
            ValueError: When no dates could be set for the 'Environment'.

        Returns:
            pd.Series[datetime]: Collection of timeseries stored in Environment.dates
        """
        if self.dates is None:
            raise ValueError("No values were assigned to dates.")
        return pd.to_datetime(self.dates["date"])

    def set_dates(
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
    ):
        """
        Set dates manually, ignoring possible dates in environmental time-series.

        Args:
            start_date (Union[str, datetime]): Start of the range dates.
            end_date (Union[str, datetime]): End of the range dates.
        """

        self.dates = self.get_dates_dataframe(start_date, end_date)

    @property
    def temp_kelvin(self) -> float:
        """
        Temperature in Kelvin.

        Returns:
            float: value representation.
        """
        if all(self.temperature.values < 100) and self.temperature is not None:
            return self.temperature + 273.15
        return self.temperature

    @property
    def temp_celsius(self) -> float:
        """
        Temperature in Celsius

        Returns:
            float: value representation.
        """
        if all(self.temperature.values > 100) and self.temperature is not None:
            return self.temperature - 273.15
        return self.temperature

    @property
    def temp_mmm(self) -> pd.DataFrame:
        """
        Temperature in Monthly mean.

        Returns:
            pd.DataFrame: value as a pandas DataFrame.
        """
        monthly_mean = self.temp_kelvin.groupby(
            [self.temp_kelvin.index.year, self.temp_kelvin.index.month]
        ).agg(["mean"])
        monthly_maximum_mean = monthly_mean.groupby(level=0).agg(["min", "max"])
        monthly_maximum_mean.columns = monthly_maximum_mean.columns.droplevel([0, 1])
        return monthly_maximum_mean

    EnvironmentValue = Union[float, list, tuple, np.ndarray, pd.DataFrame]

    def set_parameter_values(
        self, parameter: str, value: EnvironmentValue, pre_date: Optional[int] = None
    ):
        """
        Set the time-series data to a time-series, or a  value. In case :param value: is not iterable, the
        :param parameter: is assumed to be constant over time. In case :param value: is iterable, make sure its length
        complies with the simulation length.

        Included parameters:
            light                       :   incoming light-intensity [umol photons m-2 s-1]
            LAC / light_attenuation     :   light attenuation coefficient [m-1]
            temperature                 :   sea surface temperature [K]
            aragonite                   :   aragonite saturation state [-]
            storm                       :   storm category, annually [-]

        Args:
            parameter (str): Parameter to be set.
            value (EnvironmentValue): New value for the parameter.
            pre_date (Optional[int], optional): Time-series start before simulation dates [yrs]. Defaults to None.
        """

        def set_value(val):
            """Function to set  value."""
            simple_dates = self.get_dates()
            if pre_date is None:
                return pd.DataFrame({parameter: val}, index=simple_dates)

            dates = pd.date_range(
                simple_dates.iloc[0] - pd.DateOffset(years=pre_date),
                simple_dates.iloc[-1],
                freq="D",
            )
            return pd.DataFrame({parameter: val}, index=dates)

        if self.dates is None:
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
            setattr(self, parameter, set_value(value))
        elif parameter == "storm":
            years = set(self.get_dates().dt.year)
            self.storm_category = pd.DataFrame(data=value, index=years)
        else:
            msg = f"Entered parameter ({parameter}) not included. See documentation."
            raise ValueError(msg)
