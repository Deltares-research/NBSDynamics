"""
coral_model v3 - hydrodynamics

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""
import faulthandler
import sys

import numpy as np
from scipy.optimize import fsolve

from src.core.hydrodynamics.delft3d import Delft3D
from src.core.hydrodynamics.transect import Transect

faulthandler.enable()


class Hydrodynamics:
    """Interface for all hydrodynamic model modes."""

    __model = None

    _x_coordinates = None
    _y_coordinates = None
    _xy_coordinates = None
    _water_depth = None

    def __init__(self, mode):
        """
        :param mode: choice of hydrodynamic model
        :type mode: None, str
        """
        self.mode = self.set_model(mode)

    def __str__(self):
        """String-representation of Hydrodynamics."""
        return f"Coupled hydrodynamic model: {str(self.model)}\n\tmode={self.mode}"

    def __repr__(self):
        """Representation of Hydrodynamics."""
        return f"Hydrodynamics(mode={self.mode})"

    @property
    def model(self):
        """Hydrodynamic model.

        :rtype: BaseHydro
        """
        return self.__model

    def set_model(self, mode: str) -> str:
        """Function that verifies if the mode is included.

        :param mode: choice of hydrodynamic model
        :type mode: None, str
        """

        modes = ("Reef0D", "Reef1D", "Delft3D", "Transect")
        if mode not in modes:
            msg = f"{mode} not in {modes}."
            raise ValueError(msg)

        model_cls = mode
        self.__model = getattr(sys.modules[__name__], model_cls)()
        self.mode = mode
        return mode

    @property
    def space(self):
        """Space-dimension."""
        return len(self.xy_coordinates)

    @property
    def x_coordinates(self):
        """The x-coordinates of the model domain.

        :rtype: numpy.ndarray
        """
        return self.model.x

    @property
    def y_coordinates(self):
        """The y-coordinates of the model domain.

        :rtype: numpy.ndarray
        """
        return self.model.y

    @property
    def xy_coordinates(self):
        """The (x,y)-coordinates of the model domain,
        retrieved from hydrodynamic model; otherwise based on provided definition.

        :rtype: numpy.ndarray
        """
        return np.array(
            [
                [self.x_coordinates[i], self.y_coordinates[i]]
                for i in range(len(self.x_coordinates))
            ]
        )

    @property
    def water_depth(self):
        """Water depth, retrieved from hydrodynamic model; otherwise based on provided definition.

        :rtype: numpy.ndarray
        """
        return self.model.water_depth

    def set_files(self, mdu=None, config=None):
        """Set critical files of hydrodynamic model.

        :param mdu: MDU-file
        :param config: config-file

        :type mdu: str
        :type config: str
        """
        [self.__set_file(key, val) for key, val in locals().items()]

    def __set_file(self, obj, file):
        """Set file of hydrodynamic model.

        :param obj: file-object to be defined
        :param file: file

        :type obj: str
        :type file: str
        """
        if file is not None and hasattr(self.model, obj):
            setattr(self.model, obj, file)

    def set_update_intervals(self, default, storm=None):
        """Set update intervals; required for Delft3D-model.

        :param default: default update interval
        :param storm: storm update interval, defaults to None

        :type default: int
        :type storm: int, optional
        """
        self.__model.update_interval = default
        self.__model.update_interval_storm = default if storm is None else storm

        if not isinstance(self.model, Delft3D):
            print(
                f"INFO: Update intervals unused; {self.mode} does not use update intervals."
            )

    def input_check(self):
        """Check if all requested content is provided, depending on the mode chosen."""
        _ = self.xy_coordinates
        _ = self.water_depth

        if isinstance(self.model, Delft3D):
            self.input_check_extra_d3d()

    def input_check_extra_d3d(self):
        """Delft3D-specific input check."""
        files = ("mdu",)
        [self.input_check_definition(file) for file in files]

        interval_types = ("update_interval", "update_interval_storm")
        [self.input_check_definition(interval) for interval in interval_types]

    def input_check_definition(self, obj):
        """Check definition of critical object."""
        if getattr(self.model, obj) is None:
            msg = f"{obj} undefined (required for {self.mode}-mode)"
            raise ValueError(msg)

    def initiate(self):
        """Initiate hydrodynamic model."""
        self.input_check()
        self.model.initiate()

    def update(self, coral, stormcat=0):
        """Update hydrodynamic model."""
        return self.model.update(coral, stormcat)

    def finalise(self):
        """Finalise hydrodynamic model."""
        self.model.finalise()


class BaseHydro:
    """Basic, empty hydrodynamic model."""

    update_interval = None
    update_interval_storm = None

    @classmethod
    def __str__(cls):
        """String-representation of BaseHydro."""
        return cls.__name__

    @property
    def settings(self):
        """Print settings of BaseHydro-model."""
        msg = "No hydrodynamic model coupled."
        return msg

    @property
    def x(self):
        """x-coordinate(s)."""
        return None

    @property
    def y(self):
        """y-coordinate(s)."""
        return None

    def initiate(self):
        """Initiate hydrodynamic model."""
        pass

    def update(self, coral, storm=False):
        """Update hydrodynamic model.

        :param coral: coral animal
        :param storm: storm conditions, defaults to False

        :type coral: Coral
        :type storm: bool, optional
        """
        if storm:
            # max(current_vel, wave_vel)
            return None, None
        # mean(current_vel, wave_vel, wave_per)
        return None, None, None

    def finalise(self):
        """Finalise hydrodynamic model."""
        pass


class Reef0D(BaseHydro):
    """Explanatory text."""

    def __init__(self):
        super().__init__()

    @property
    def settings(self):
        """Print settings of Reef0D-model."""
        msg = f"Not yet implemented."
        return msg


class Reef1D(BaseHydro):
    """Simplified one-dimensional hydrodynamic model over a (coral) reef."""

    # TODO: Complete the one-dimensional hydrodynamic model

    def __init__(self):
        """Internal 1D hydrodynamic model for order-of-magnitude calculations on the hydrodynamic conditions on a coral
        reef, where both flow and waves are included.
        """
        super().__init__()

        self.bath = None
        self.Hs = None
        self.Tp = None
        self.dx = None

        # self.z = np.zeros(self.space)

        self._diameter = None
        self._height = None
        self._density = None

    def __repr__(self):
        msg = (
            f"Reef1D(bathymetry={self.bath}, wave_height={self.Hs}, "
            f"wave_period={self.Tp})"
        )
        return msg

    @property
    def settings(self):
        """Print settings of Reef1D-model."""
        msg = (
            f"One-dimensional simple hydrodynamic model to simulate the "
            f"hydrodynamics on a (coral) reef with the following settings:"
            f"\n\tBathymetric cross-shore data : {type(self.bath).__name__}"
            f"\n\t\trange [m]  : {min(self.bath)}-{max(self.bath)}"
            f"\n\t\tlength [m] : {self.space * self.dx}"
            f"\n\tSignificant wave height [m]  : {self.Hs}"
            f"\n\tPeak wave period [s]         : {self.Tp}"
        )
        return msg

    @property
    def space(self):
        return len(self.bath)

    @property
    def x(self):
        return np.arange(0, self.space, self.dx)

    @property
    def y(self):
        return 0

    @property
    def vel_wave(self):
        return 0

    @property
    def vel_curr_mn(self):
        return 0

    @property
    def vel_curr_mx(self):
        return 0

    @property
    def per_wav(self):
        return self.Tp

    @property
    def water_level(self):
        return 0

    @property
    def depth(self):
        return self.bath + self.water_level

    @property
    def can_dia(self):
        return self._diameter

    @can_dia.setter
    def can_dia(self, canopy_diameter):
        self._diameter = canopy_diameter

    @property
    def can_height(self):
        return self._height

    @can_height.setter
    def can_height(self, canopy_height):
        self._height = canopy_height

    @property
    def can_den(self):
        return self._density

    @can_den.setter
    def can_den(self, canopy_density):
        self._density = canopy_density

    @staticmethod
    def dispersion(wave_length, wave_period, depth, grav_acc):
        """Dispersion relation to determine the wave length based on the
        wave period.
        """
        func = wave_length - ((grav_acc * wave_period ** 2) / (2 * np.pi)) * np.tanh(
            2 * np.pi * depth / wave_length
        )
        return func

    @property
    def wave_length(self):
        """Solve the dispersion relation to retrieve the wave length."""
        L0 = 9.81 * self.per_wav ** 2
        L = np.zeros(len(self.depth))
        for i, h in enumerate(self.depth):
            if h > 0:
                L[i] = fsolve(self.dispersion, L0, args=(self.per_wav, h, 9.81))
        return L

    @property
    def wave_frequency(self):
        return 2 * np.pi / self.per_wav

    @property
    def wave_number(self):
        k = np.zeros(len(self.wave_length))
        k[self.wave_length > 0] = 2 * np.pi / self.wave_length[self.wave_length > 0]
        return k

    @property
    def wave_celerity(self):
        return self.wave_length / self.per_wav

    @property
    def group_celerity(self):
        n = 0.5 * (
            1
            + (2 * self.wave_number * self.depth)
            / (np.sinh(self.wave_number * self.depth))
        )
        return n * self.wave_celerity
