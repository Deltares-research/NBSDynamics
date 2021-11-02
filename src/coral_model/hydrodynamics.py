"""
coral_model v3 - hydrodynamics

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""
import faulthandler
import os
import sys

from bmi.wrapper import BMIWrapper
import numpy as np
from scipy.optimize import fsolve

faulthandler.enable()

# class Hydrodynamics:
#     """Interface for all hydrodynamic model modes."""

#     __model = None

#     _x_coordinates = None
#     _y_coordinates = None
#     _xy_coordinates = None
#     _water_depth = None

#     def __init__(self, mode):
#         """
#         :param mode: choice of hydrodynamic model
#         :type mode: None, str
#         """
#         self.mode = self.set_model(mode)

#     def __str__(self):
#         """String-representation of Hydrodynamics."""
#         return f'Coupled hydrodynamic model: {str(self.model)}\n\tmode={self.mode}'

#     def __repr__(self):
#         """Representation of Hydrodynamics."""
#         return f'Hydrodynamics(mode={self.mode})'

#     @property
#     def model(self):
#         """Hydrodynamic model.

#         :rtype: BaseHydro
#         """
#         return self.__model

#     def set_model(self, mode):
#         """Function that verifies if the mode is included.

#         :param mode: choice of hydrodynamic model
#         :type mode: None, str
#         """

#         modes = ('Reef0D', 'Reef1D', 'Delft3D','Transect')
#         if mode not in modes:
#             msg = f'{mode} not in {modes}.'
#             raise ValueError(msg)

#         model_cls = mode
#         self.__model = getattr(sys.modules[__name__], model_cls)()
#         return mode

#     @property
#     def space(self):
#         """Space-dimension."""
#         return len(self.xy_coordinates)

#     @property
#     def x_coordinates(self):
#         """The x-coordinates of the model domain.

#         :rtype: numpy.ndarray
#         """
#         return self.model.x

#     @property
#     def y_coordinates(self):
#         """The y-coordinates of the model domain.

#         :rtype: numpy.ndarray
#         """
#         return self.model.y

#     @property
#     def xy_coordinates(self):
#         """The (x,y)-coordinates of the model domain,
#         retrieved from hydrodynamic model; otherwise based on provided definition.

#         :rtype: numpy.ndarray
#         """
#         return np.array([
#                 [self.x_coordinates[i], self.y_coordinates[i]] for i in range(len(self.x_coordinates))
#             ])

#     @property
#     def water_depth(self):
#         """Water depth, retrieved from hydrodynamic model; otherwise based on provided definition.

#         :rtype: numpy.ndarray
#         """
#         return self.model.water_depth

#     def set_files(self, mdu=None, config=None):
#         """Set critical files of hydrodynamic model.

#         :param mdu: MDU-file
#         :param config: config-file

#         :type mdu: str
#         :type config: str
#         """
#         [self.__set_file(key, val) for key, val in locals().items()]

#     def __set_file(self, obj, file):
#         """Set file of hydrodynamic model.

#         :param obj: file-object to be defined
#         :param file: file

#         :type obj: str
#         :type file: str
#         """
#         if file is not None and hasattr(self.model, obj):
#             setattr(self.model, obj, file)

#     def set_update_intervals(self, default, storm=None):
#         """Set update intervals; required for Delft3D-model.

#         :param default: default update interval
#         :param storm: storm update interval, defaults to None

#         :type default: int
#         :type storm: int, optional
#         """
#         self.__model.update_interval = default
#         self.__model.update_interval_storm = default if storm is None else storm

#         if not isinstance(self.model, Delft3D):
#             print(f'INFO: Update intervals unused; {self.mode} does not use update intervals.')

#     def input_check(self):
#         """Check if all requested content is provided, depending on the mode chosen."""
#         _ = self.xy_coordinates
#         _ = self.water_depth

#         if isinstance(self.model, Delft3D):
#             self.input_check_extra_d3d()

#     def input_check_extra_d3d(self):
#         """Delft3D-specific input check."""
#         files = ('mdu',)
#         [self.input_check_definition(file) for file in files]

#         interval_types = ('update_interval', 'update_interval_storm')
#         [self.input_check_definition(interval) for interval in interval_types]

#     def input_check_definition(self, obj):
#         """Check definition of critical object."""
#         if getattr(self.model, obj) is None:
#             msg = f'{obj} undefined (required for {self.mode}-mode)'
#             raise ValueError(msg)

#     def initiate(self):
#         """Initiate hydrodynamic model."""
#         self.input_check()
#         self.model.initiate()

#     def update(self, coral, stormcat=0):
#         """Update hydrodynamic model."""
#         return self.model.update(coral, stormcat)

#     def finalise(self):
#         """Finalise hydrodynamic model."""
#         self.model.finalise()


# class BaseHydro:
#     """Basic, empty hydrodynamic model."""

#     update_interval = None
#     update_interval_storm = None

#     @classmethod
#     def __str__(cls):
#         """String-representation of BaseHydro."""
#         return cls.__name__

#     @property
#     def settings(self):
#         """Print settings of BaseHydro-model."""
#         msg = 'No hydrodynamic model coupled.'
#         return msg

#     @property
#     def x(self):
#         """x-coordinate(s)."""
#         return None

#     @property
#     def y(self):
#         """y-coordinate(s)."""
#         return None

#     def initiate(self):
#         """Initiate hydrodynamic model."""

#     def update(self, coral, storm=False):
#         """Update hydrodynamic model.

#         :param coral: coral animal
#         :param storm: storm conditions, defaults to False

#         :type coral: Coral
#         :type storm: bool, optional
#         """
#         if storm:
#             # max(current_vel, wave_vel)
#             return None, None
#         # mean(current_vel, wave_vel, wave_per)
#         return None, None, None

#     def finalise(self):
#         """Finalise hydrodynamic model."""


# class Reef0D(BaseHydro):
#     """Explanatory text."""

#     def __init__(self):
#         super().__init__()

#     @property
#     def settings(self):
#         """Print settings of Reef0D-model."""
#         msg = f'Not yet implemented.'
#         return msg

#     def initiate(self):
#         pass

#     def update(self, coral, storm=False):
#         pass

#     def finalise(self):
#         pass


# class Reef1D(BaseHydro):
#     """Simplified one-dimensional hydrodynamic model over a (coral) reef."""
#     # TODO: Complete the one-dimensional hydrodynamic model

#     def __init__(self):
#         """Internal 1D hydrodynamic model for order-of-magnitude calculations on the hydrodynamic conditions on a coral
#         reef, where both flow and waves are included.
#         """
#         super().__init__()

#         self.bath = None
#         self.Hs = None
#         self.Tp = None
#         self.dx = None

#         # self.z = np.zeros(self.space)

#         self._diameter = None
#         self._height = None
#         self._density = None

#     def __repr__(self):
#         msg = f'Reef1D(bathymetry={self.bath}, wave_height={self.Hs}, ' \
#             f'wave_period={self.Tp})'
#         return msg

#     @property
#     def settings(self):
#         """Print settings of Reef1D-model."""
#         msg = f'One-dimensional simple hydrodynamic model to simulate the ' \
#             f'hydrodynamics on a (coral) reef with the following settings:' \
#             f'\n\tBathymetric cross-shore data : {type(self.bath).__name__}' \
#             f'\n\t\trange [m]  : {min(self.bath)}-{max(self.bath)}' \
#             f'\n\t\tlength [m] : {self.space * self.dx}' \
#             f'\n\tSignificant wave height [m]  : {self.Hs}' \
#             f'\n\tPeak wave period [s]         : {self.Tp}'
#         return msg

#     @property
#     def space(self):
#         return len(self.bath)

#     @property
#     def x(self):
#         return np.arange(0, self.space, self.dx)

#     @property
#     def y(self):
#         return 0

#     @property
#     def vel_wave(self):
#         return 0

#     @property
#     def vel_curr_mn(self):
#         return 0

#     @property
#     def vel_curr_mx(self):
#         return 0

#     @property
#     def per_wav(self):
#         return self.Tp

#     @property
#     def water_level(self):
#         return 0

#     @property
#     def depth(self):
#         return self.bath + self.water_level

#     @property
#     def can_dia(self):
#         return self._diameter

#     @can_dia.setter
#     def can_dia(self, canopy_diameter):
#         self._diameter = canopy_diameter

#     @property
#     def can_height(self):
#         return self._height

#     @can_height.setter
#     def can_height(self, canopy_height):
#         self._height = canopy_height

#     @property
#     def can_den(self):
#         return self._density

#     @can_den.setter
#     def can_den(self, canopy_density):
#         self._density = canopy_density

#     @staticmethod
#     def dispersion(wave_length, wave_period, depth, grav_acc):
#         """Dispersion relation to determine the wave length based on the
#         wave period.
#         """
#         func = wave_length - ((grav_acc * wave_period ** 2) / (2 * np.pi)) * \
#             np.tanh(2 * np.pi * depth / wave_length)
#         return func

#     @property
#     def wave_length(self):
#         """Solve the dispersion relation to retrieve the wave length."""
#         L0 = 9.81 * self.per_wav ** 2
#         L = np.zeros(len(self.depth))
#         for i, h in enumerate(self.depth):
#             if h > 0:
#                 L[i] = fsolve(self.dispersion, L0, args=(self.per_wav, h, 9.81))
#         return L

#     @property
#     def wave_frequency(self):
#         return 2 * np.pi / self.per_wav

#     @property
#     def wave_number(self):
#         k = np.zeros(len(self.wave_length))
#         k[self.wave_length > 0] = 2 * np.pi / self.wave_length[
#             self.wave_length > 0]
#         return k

#     @property
#     def wave_celerity(self):
#         return self.wave_length / self.per_wav

#     @property
#     def group_celerity(self):
#         n = .5 * (1 + (2 * self.wave_number * self.depth) /
#                   (np.sinh(self.wave_number * self.depth)))
#         return n * self.wave_celerity

#     def initiate(self):
#         pass

#     def update(self, coral, storm=False):
#         pass

#     def finalise(self):
#         pass


class Delft3D:
    """Coupling of coral_model to Delft3D using the BMI wrapper."""

    _home = None
    _dflow_dir = None
    _dimr_dir = None

    _working_dir = None
    _mdu = None
    _config = None

    _model_fm = None
    _model_dimr = None

    _space = None
    _water_depth = None

    _x_coordinates = None
    _y_coordinates = None
    _xy_coordinates = None

    def __init__(self):

        self.time_step = None

    def __repr__(self):
        msg = f"Delft3D()"
        return msg

    @property
    def settings(self):
        """Print settings of Delft3D-model."""
        if self.config:
            incl = f"DFlow- and DWaves-modules"
            files = (
                f"\n\tDFlow file         : {self.mdu}"
                f"\n\tConfiguration file : {self.config}"
            )
        else:
            incl = f"DFlow-module"
            files = f"\n\tDFlow file         : {self.mdu}"

        msg = (
            f"Coupling with Delft3D model (incl. {incl}) with the following settings:"
            f"\n\tDelft3D home dir.  : {self.d3d_home}"
            f"{files}"
        )
        return msg

    @property
    def d3d_home(self):
        """Delft3D home directory.

        :rtype: str
        """
        return self._home

    @d3d_home.setter
    def d3d_home(self, folder):
        """
        :param folder: Delft3D home directory
        :type folder: str
        """
        self._home = folder

    @property
    def working_dir(self):
        """Model working directory."""
        return self._working_dir

    @working_dir.setter
    def working_dir(self, folder):
        """
        :param folder: working directory
        :type folder:  str
        """
        self._working_dir = folder

    @property
    def dflow_dir(self):
        """Directory to DFlow-ddl."""
        self._dflow_dir = os.path.join(self.d3d_home, "dflowfm", "bin", "dflowfm")
        return self._dflow_dir

    @property
    def dimr_dir(self):
        """Directory to DIMR-dll."""
        self._dimr_dir = os.path.join(self.d3d_home, "dimr", "bin", "dimr_dll")
        return self._dimr_dir

    @property
    def mdu(self):
        """Delft3D's MDU-file.

        :rtype: str
        """
        return self._mdu

    @mdu.setter
    def mdu(self, file_dir):
        """
        :param file_dir: file directory of MDU-file
        :type file_dir: str
        """
        self._mdu = os.path.join(self.working_dir, file_dir)

    @property
    def config(self):
        """Delft3D's config-file.

        :rtype: str
        """
        return self._config

    @config.setter
    def config(self, file_dir):
        """
        :param file_dir: file directory of config-file
        :type file_dir: str, list, tuple
        """
        self._config = os.path.join(self.working_dir, file_dir)

    @property
    def model(self):
        """Main model-object."""
        return self.model_dimr if self.config else self.model_fm

    @property
    def model_fm(self):
        """Deflt3D-FM model-object."""
        return self._model_fm

    @property
    def model_dimr(self):
        """Delft3D DIMR model-object."""
        return self._model_dimr

    def environment(self):
        """Set Python environment to include Delft3D-code."""
        dirs = [
            os.path.join(self.d3d_home, "share", "bin"),
            os.path.join(self.d3d_home, "dflowfm", "bin"),
        ]
        if self.config:
            dirs.extend(
                [
                    os.path.join(self.d3d_home, "dimr", "bin"),
                    os.path.join(self.d3d_home, "dwaves", "bin"),
                    os.path.join(self.d3d_home, "esmf", "scripts"),
                    os.path.join(self.d3d_home, "swan", "scripts"),
                ]
            )

        env = ";".join(dirs)
        os.environ["PATH"] = env

        print(f'\nEnvironment "PATH":')
        [print(f"\t{path}") for path in dirs]

    def input_check(self):
        """Check if all requested content is provided"""

        self.input_check_definition("xy_coordinates")
        self.input_check_definition("water_depth")

        files = ("mdu",)
        [self.input_check_definition(file) for file in files]

        interval_types = ("update_interval", "update_interval_storm")
        [self.input_check_definition(interval) for interval in interval_types]

    def input_check_definition(self, obj):
        """Check definition of critical object."""
        if getattr(self.model, obj) is None:
            msg = f"{obj} undefined (required for Delft3D coupling)"
            raise ValueError(msg)

    def get_variable(self, variable):
        """Get variable from DFlow-model.

        :param variable: variable to get
        :type variable: str
        """
        return self._model_fm.get_var(variable)

    def set_variable(self, variable, value):
        """Set variable to DFlow-model.

        :param variable: variable to set
        :param value: value of variable

        :type variable: str
        :type value: float, list, tuple, numpy.ndarray
        """
        self._model_fm.set_var(variable, value)

    @property
    def space(self):
        """Number of non-boundary boxes; i.e. within-domain boxes."""
        self._space = self.get_variable("ndxi") if self._space is None else self._space
        return self._space.item()

    @property
    def x_coordinates(self):
        """Center of gravity's x-coordinates as part of `space`."""
        self._x_coordinates = (
            self.get_variable("xzw")[range(self.space)]
            if self._x_coordinates is None
            else self._x_coordinates
        )
        return self._x_coordinates

    @property
    def y_coordinates(self):
        """Center of gravity's y-coodinates as part of `space`."""
        self._y_coordinates = (
            self.get_variable("yzw")[range(self.space)]
            if self._y_coordinates is None
            else self._y_coordinates
        )
        return self._y_coordinates

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
        """Water depth."""
        if self.time_step is None:
            self.time_step = self.get_var("is_dtint")
        if self._water_depth is None:
            return (
                self.get_variable("is_sumvalsnd")[range(self.space), 2] / self.time_step
            )
        else:
            return self._water_depth

    def set_update_intervals(self, default, storm=None):
        """Set update intervals

        :param default: default update interval
        :param storm: storm update interval, defaults to None

        :type default: int
        :type storm: int, optional
        """
        self.update_interval = default
        self.update_interval_storm = default if storm is None else storm

    def reset_counters(self):
        """Reset properties for next model update."""
        sums = self.get_variable("is_sumvalsnd")
        sums.fill(0.0)
        self.set_variable("is_sumvalsnd", sums)

        maxs = self.get_variable("is_maxvalsnd")
        maxs.fill(0.0)
        self.set_variable("is_maxvalsnd", maxs)

    def set_morphology(self, coral):
        """Set morphological dimensions to Delft3D-model.

        :param coral: coral animal
        :type coral: Coral
        """
        self.set_variable("rnveg", coral.as_vegetation_density)
        self.set_variable("diaveg", coral.dc_rep)
        self.set_variable("stemheight", coral.hc)

    def get_mean_hydrodynamics(self):
        """Get hydrodynamic results; mean values."""
        if self.time_step is None:
            self.time_step = self.get_variable("is_dtint")
        current_vel = (
            self.get_variable("is_sumvalsnd")[range(self.space), 1] / self.time_step
        )
        wave_vel = self.get_variable("Uorb")[range(self.space)]
        wave_per = self.get_variable("twav")[range(self.space)]
        return current_vel, wave_vel, wave_per

    def get_max_hydrodynamics(self):
        """Get hydrodynamic results; max. values."""
        current_vel = self.get_variable("is_maxvalsnd")[range(self.space), 1]
        wave_vel = self.get_variable("Uorb")[range(self.space)]
        wave_per = self.get_variable("twav")[range(self.space)]
        return current_vel, wave_vel, wave_per

    def initiate(self):
        """Initialize the working model."""
        self.environment()
        self._model_fm = BMIWrapper(engine=self.dflow_dir, configfile=self.mdu)
        if self.config:
            self._model_dimr = BMIWrapper(engine=self.dimr_dir, configfile=self.config)
        self.model.initialize()  # if self.model_dimr is None else self.model_dimr.initialize()

    def update(self, coral, stormcat=0):
        """Update the Delft3D-model."""
        self.time_step = (
            self.update_interval_storm if stormcat > 0 else self.update_interval
        )
        self.reset_counters()
        self.model.update(self.time_step)

        return (
            self.get_max_hydrodynamics()
            if stormcat > 0
            else self.get_mean_hydrodynamics()
        )

    def finalise(self):
        """Finalize the working model."""
        self.model.finalize()


class Transect:
    """Simple 1D depth transect with imposed currents and waves"""

    _home = None

    _working_dir = None
    _mdu = None
    _config = None

    _space = None
    _x_coordinates = None
    _y_coordinates = None
    _water_depth = None

    def __init__(self):
        self.time_step = None

    def __repr__(self):
        msg = f"Transect()"
        return msg

    @property
    def settings(self):
        """Print settings of simple transect imposed forcing."""
        files = (
            f"\n\tTransect config file  : {self.config}"
            f"\n\tTransect forcings file : {self.mdu}"
        )
        msg = (
            f"1D schematic cross-shore transect with forced hydrodynamics: "
            f"\n\tTransect model working dir.  : {self.working_dir}"
            f"{files}"
        )
        return msg

    @property
    def working_dir(self):
        """Model working directory."""
        return self._working_dir

    @working_dir.setter
    def working_dir(self, folder):
        """
        :param folder: working directory
        :type folder:  str
        """
        self._working_dir = folder

    @property
    def mdu(self):
        """Delft3D's MDU-file.

        :rtype: str
        """
        return self._mdu

    @mdu.setter
    def mdu(self, file_dir):
        """
        :param file_dir: file directory of MDU-file
        :type file_dir: str
        """
        self._mdu = os.path.join(self.working_dir, file_dir)

    @property
    def config(self):
        """Transect's config-file.

        :rtype: str
        """
        return self._config

    @config.setter
    def config(self, file_dir):
        """
        :param file_dir: file directory of config-file
        :type file_dir: str, list, tuple
        """
        self._config = os.path.join(self.working_dir, file_dir)

    def input_check(self):
        """Check if all requested content is provided"""

        self.input_check_definition("xy_coordinates")
        self.input_check_definition("water_depth")

        files = ("mdu", "config")
        [self.input_check_definition(file) for file in files]

    def input_check_definition(self, obj):
        """Check definition of critical object."""
        if getattr(self.model, obj) is None:
            msg = f"{obj} undefined (required for Transect)"
            raise ValueError(msg)

    @property
    def space(self):
        """Number of non-boundary boxes; i.e. within-domain boxes."""
        return len(self._x_coordinates)

    @property
    def x_coordinates(self):
        """Center of gravity's x-coordinates as part of `space`."""
        return self._x_coordinates

    @property
    def y_coordinates(self):
        """Center of gravity's y-coodinates as part of `space`."""
        return self._y_coordinates

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
        """Water depth."""
        return self._water_depth

    @property
    def outpoint(self):
        """coordinates where his output is desired"""
        return self._outpoint

    def reset_counters(self):
        """Reset properties for next model update."""

    def set_morphology(self, coral):
        """Set morphological dimensions to Delft3D-model.

        :param coral: coral animal
        :type coral: Coral
        """

    def initiate(self):
        """Initialize the working model.
        In this case, read the spatial configuration and the forcings
        from files. Set the computing environment.
        """
        csv = np.genfromtxt(self.config, delimiter=",", skip_header=1)
        self._x_coordinates = csv[:, 0]
        self._y_coordinates = csv[:, 1]
        self._water_depth = csv[:, 2]
        self._outpoint = csv[:, 3] == 1

        self.forcings = np.genfromtxt(self.mdu, delimiter=",", skip_header=1)
        self.stormcat = self.forcings[:, 0]
        self.return_period = self.forcings[:, 1]
        self.wave_height = self.forcings[:, 2]
        self.wave_period = self.forcings[:, 3]
        self.wave_angle = self.forcings[:, 4]
        self.max_curr_vel = self.forcings[:, 5]

    def update(self, coral, stormcat=0):
        """Update the model, which is just knowing the waves"""
        # Not sure if this method is currently being used, but
        # just in case we better make it point to the lower one
        # to avoid code duplication. Either way, the coral parameter
        # was not being used.
        return self.update_orbital(stormcat)

    def update_orbital(self, stormcat=0):
        """Update the model, which is just knowing the waves"""
        mean_current_vel = 0
        if stormcat in [0, 1, 2, 3]:
            Hs = self.wave_height[stormcat]
            T = self.wave_period[stormcat]
            max_current_vel = self.max_curr_vel[stormcat]
            h = self._water_depth
            wave_vel = (
                Hs
                / 4
                * np.sqrt(9.81 / h)
                * np.exp(-np.power((3.65 / T * np.sqrt(h / 9.81)), 2.1))
            )
        else:
            msg = f"stormcat = {stormcat}, must be either 0,1,2,3"
            raise ValueError(msg)
        if stormcat == 0:
            return mean_current_vel, wave_vel, T
        else:
            return max_current_vel, wave_vel, T

    def finalise(self):
        """Finalize the working model."""
        pass
