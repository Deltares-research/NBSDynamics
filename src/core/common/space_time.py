"""
coral_model - utils
veg_model - utils

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""

# TODO: Restructure all utils-related files, methods, and methods.
from typing import Optional, Tuple, Union

import numpy as np
from pandas import DataFrame

from src.biota_models.coral.model.coral_only import CoralOnly
from src.biota_models.vegetation.model.veg_only import VegOnly


class SpaceTime:
    """Spacetime-object, which validates the definition of the spacetime dimensions."""

    __spacetime = None

    def __init__(self, spacetime: Optional[Tuple] = None):
        """
        :param spacetime: spacetime dimensions, defaults to None
        :type spacetime: None, tuple, optional
        """
        if spacetime is not None:
            self.spacetime = spacetime

        self.set_coral_only(self.spacetime)
        self.set_veg_only(self.spacetime)

    def __repr__(self):
        """Development representation."""
        return f"SpaceTime({self.__spacetime})"

    def __str__(self):
        """Print representation."""
        return str(self.spacetime)

    @property
    def spacetime(self):
        """Spacetime dimensions.

        :rtype: tuple
        """
        if self.__spacetime is None:
            return 1, 1
        return self.__spacetime

    @spacetime.setter
    def spacetime(self, space_time: Union[Tuple, list, np.ndarray]):
        """
        :param space_time: spacetime dimensions
        :type space_time: tuple, list, numpy.ndarray
        """
        if not isinstance(space_time, (tuple, list, np.ndarray)):
            msg = f"spacetime must be of type tuple, {type(space_time)} is given."
            raise TypeError(msg)

        if not len(space_time) == 2:
            msg = f"spacetime must be of size 2, {len(space_time)} is given."
            raise ValueError(msg)

        if not all(isinstance(dim, int) for dim in space_time):
            msg = f"spacetime must consist of integers only, {[type(dim) for dim in space_time]} is given."
            raise TypeError(msg)

        self.__spacetime = tuple(space_time)
        self.set_coral_only(tuple(space_time))
        self.set_veg_only(tuple(space_time))

    @property
    def space(self) -> int:
        """Space dimension.

        :rtype: int
        """
        return self.spacetime[0]

    @space.setter
    def space(self, x: int):
        """
        :param x: space dimension
        :type x: int
        """
        self.spacetime = (x, self.time)

    @property
    def time(self) -> int:
        """Time dimension.

        :rtype: int
        """
        return self.spacetime[1]

    @time.setter
    def time(self, t: int):
        """
        :param t: time dimension
        :type t: int
        """
        self.spacetime = (self.space, t)

    # TODO: Refactor to a private method
    def set_coral_only(self, spacetime: Tuple):
        """Automatically set the spacetime dimensions for the CoralOnly-class.

        :param spacetime: spacetime dimension
        :type spacetime: tuple
        """
        CoralOnly.spacetime = spacetime

    def set_veg_only(self, spacetime: Tuple):
        """Automatically set the spacetime dimensions for the VegOnly-class.

        :param spacetime: spacetime dimension
        :type spacetime: tuple
        """
        VegOnly.spacetime = spacetime


class DataReshape(SpaceTime):
    """Reshape data to create a spacetime matrix."""

    def __init__(self, spacetime: Optional[Tuple] = None):
        """
        :param spacetime: spacetime dimensions, defaults to None
        :type spacetime: None, tuple, optional
        """
        super().__init__(spacetime=spacetime)

    def variable2matrix(
        self, variable: Union[float, int, list, Tuple, np.ndarray], dimension: str
    ):
        """Transform variable to matrix.

        :param variable: variable to be transformed
        :param dimension: dimension of :param variable:

        :type variable: float, int, list, tuple, numpy.ndarray
        :type dimension: str

        :return: variable as matrix in space-time
        :rtype: numpy.ndarray
        """
        # # input check
        # dimension-type
        dimensions = ("space", "time")
        if dimension not in dimensions:
            msg = f"{dimension} not in {dimensions}."
            raise ValueError(msg)

        # dimension-value
        variable = self.variable2array(variable)
        self.dimension_value(variable, dimension)

        # # transformation
        if dimension == "space":
            return np.tile(variable, (self.time, 1)).transpose()
        elif dimension == "time":
            return np.tile(variable, (self.space, 1))

    def dimension_value(self, variable: Union[list, tuple, np.ndarray], dimension: str):
        """Check consistency between variable's dimensions and the defined spacetime dimensions.

        :param variable: variable to be checked
        :param dimension: dimension under consideration

        :type variable: list, tuple, numpy.ndarray
        :type dimension: str
        """
        try:
            _ = len(variable)
        except TypeError:
            variable = [variable]

        if not len(variable) == getattr(self, dimension):
            msg = f"Incorrect variable size, {len(variable)} =/= {getattr(self, dimension)}."
            raise ValueError(msg)

    @staticmethod
    def variable2array(variable: Union[float, int, list, np.ndarray]):
        """ "Transform variable to numpy.array (if float or string).

        :param variable: variable to be transformed
        :type variable: float, int, list, numpy.ndarray

        :return: variable as array
        :rtype: numpy.ndarray
        """
        if isinstance(variable, str):
            msg = f"Variable cannot be of {type(variable)}."
            raise NotImplementedError(msg)
        elif isinstance(variable, (float, int)):
            return np.array([float(variable)])
        elif isinstance(variable, (list, tuple)):
            return np.array(variable)
        elif isinstance(variable, np.ndarray) and not variable.shape:
            return np.array([variable])
        return variable

    def matrix2array(
        self, matrix: np.ndarray, dimension: str, conversion: Optional[str] = None
    ):
        """Transform matrix to array.

        :param matrix: variable as matrix in spacetime
        :param dimension: dimension to convert matrix to
        :param conversion: how to convert the matrix to an array, defaults to None
            None    :   take the last value
            'mean'  :   take the mean value
            'max'   :   take the maximum value
            'min'   :   take the minimum value
            'sum'   :   take the summation

        :type matrix: numpy.ndarray
        :type dimension: str
        :type conversion: None, str, optional

        :return: variable as array
        :rtype: numpy.ndarray
        """
        # # input check
        # dimension-type
        dimensions = ("space", "time")
        if dimension not in dimensions:
            msg = f"{dimension} not in {dimensions}."
            raise ValueError(msg)

        # input as numpy.array
        matrix = np.array(matrix)

        # dimension-value
        if (
            not matrix.shape == self.spacetime
            and not matrix.shape[:2] == self.spacetime
        ):
            msg = (
                f"Matrix-shape does not correspond with spacetime-dimensions:"
                f"\n{matrix.shape} =/= {self.spacetime}"
            )
            raise ValueError(msg)

        # conversion-strategy
        conversions = (None, "mean", "max", "min", "sum")
        if conversion not in conversions:
            msg = f"{conversion} not in {conversions}."
            raise ValueError(msg)

        # # transformation
        # last position
        if conversion is None:
            if dimension == "space":
                return matrix[:, -1]
            elif dimension == "time":
                return matrix[-1, :]

        # conversion
        if dimension == "space":
            return getattr(matrix, conversion)(axis=1)
        elif dimension == "time":
            return getattr(matrix, conversion)(axis=0)


def time_series_year(time_series: DataFrame, year: int):
    """Extract a section of the time-series based on the year.

    :param time_series: time-series to be extracted
    :param year: year to be extracted

    :type time_series: pandas.DataFrame
    :type year: int
    """
    return time_series[time_series.index.year == year].values.transpose()[0]
