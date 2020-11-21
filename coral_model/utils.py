"""
coral_model v3 - utils

@author: Gijs G. Hendrickx
"""

import numpy as np


class SpaceTime:
    """Spacetime-object, which validates the definition of the spacetime dimensions."""

    __spacetime = None

    def __init__(self, spacetime=None):
        """
        :param spacetime: spacetime dimensions, defaults to None
        :type spacetime: None, tuple, optional
        """
        if spacetime is not None:
            self.spacetime = spacetime

    def __repr__(self):
        """Development representation."""
        return f'SpaceTime({self.__spacetime})'

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
    def spacetime(self, space_time):
        """
        :param space_time: spacetime dimensions
        :type space_time: tuple, list, numpy.ndarray
        """
        if not isinstance(space_time, (tuple, list, np.ndarray)):
            msg = f'spacetime must be of type tuple, {type(space_time)} is given.'
            raise TypeError(msg)

        if not len(space_time) == 2:
            msg = f'spacetime must be of size 2, {len(space_time)} is given.'
            raise ValueError(msg)

        if not all(isinstance(dim, int) for dim in space_time):
            msg = f'spacetime must consist of integers only, {[type(dim) for dim in space_time]} is given.'
            raise TypeError(msg)

        self.__spacetime = tuple(space_time)

    @property
    def space(self):
        """Space dimension.

        :rtype: int
        """
        return self.spacetime[0]

    @space.setter
    def space(self, x):
        """
        :param x: space dimension
        :type x: int
        """
        self.spacetime = (x, self.time)

    @property
    def time(self):
        """Time dimension.

        :rtype: int
        """
        return self.spacetime[1]

    @time.setter
    def time(self, t):
        """
        :param t: time dimension
        :type t: int
        """
        self.spacetime = (self.space, t)


class DataReshape(SpaceTime):
    """Reshape data to create a spacetime matrix."""

    def __init__(self, spacetime=None):
        """
        :param spacetime: spacetime dimensions, defaults to None
        :type spacetime: None, tuple, optional
        """
        super().__init__(spacetime=spacetime)
    
    def variable2matrix(self, variable, dimension):
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
        dimensions = ('space', 'time')
        if dimension not in dimensions:
            msg = f'{dimension} not in {dimensions}.'
            raise ValueError(msg)

        # dimension-value
        variable = self.variable2array(variable)
        self.dimension_value(variable, dimension)

        # # transformation
        if dimension == 'space':
            return np.tile(variable, (self.time, 1)).transpose()
        elif dimension == 'time':
            return np.tile(variable, (self.space, 1))

    def dimension_value(self, variable, dimension):
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
            msg = f'Incorrect variable size, {len(variable)} =/= {getattr(self, dimension)}.'
            raise ValueError(msg)

    @staticmethod
    def variable2array(variable):
        """"Transform variable to numpy.array (if float or string).
        
        :param variable: variable to be transformed
        :type variable: float, int, list, numpy.ndarray

        :return: variable as array
        :rtype: numpy.ndarray
        """
        if isinstance(variable, str):
            msg = f'Variable cannot be of {type(variable)}.'
            raise NotImplementedError(msg)
        elif isinstance(variable, (float, int)):
            return np.array([float(variable)])
        elif isinstance(variable, (list, tuple)):
            return np.array(variable)
        elif isinstance(variable, np.ndarray) and not variable.shape:
            return np.array([variable])
        return variable

    def matrix2array(self, matrix, dimension, conversion=None):
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
        dimensions = ('space', 'time')
        if dimension not in dimensions:
            msg = f'{dimension} not in {dimensions}.'
            raise ValueError(msg)

        # input as numpy.array
        matrix = np.array(matrix)

        # dimension-value
        if not matrix.shape == self.spacetime:
            if not matrix.shape[:2] == self.spacetime:
                msg = f'Matrix-shape does not correspond with spacetime-dimensions:' \
                      f'\n{matrix.shape} =/= {self.spacetime}'
                raise ValueError(msg)

        # conversion-strategy
        conversions = (None, 'mean', 'max', 'min', 'sum')
        if conversion not in conversions:
            msg = f'{conversion} not in {conversions}.'
            raise ValueError(msg)

        # # transformation
        # last position
        if conversion is None:
            if dimension == 'space':
                return matrix[:, -1]
            elif dimension == 'time':
                return matrix[-1, :]

        # conversion
        if dimension == 'space':
            return getattr(matrix, conversion)(axis=1)
        elif dimension == 'time':
            return getattr(matrix, conversion)(axis=0)


def coral_only_function(coral, function, args, no_cover_value=0):
    """Only execute the function when there is coral cover.

    :param coral: coral object
    :param function: function to be executed
    :param args: input arguments of the function
    :param no_cover_value: default value in absence of coral cover

    :type coral: Coral
    :type args: tuple
    :type no_cover_value: float, optional
    """
    try:
        size = len(coral.cover)
    except TypeError:
        size = 1

    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, (float, int)) or (isinstance(arg, np.ndarray) and not arg.shape):
            args[i] = np.repeat(arg, size)
        elif not len(arg) == size:
            msg = f'Sizes do not match up, {len(arg)} =/= {size}.'
            raise ValueError(msg)

    output = no_cover_value * np.ones(size)
    output[coral.cover > 0] = function(*[
        arg[coral.cover > 0] for arg in args
    ])
    return output

# TODO: Include methods on writing the output files here in "utils.py"
