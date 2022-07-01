import abc
import faulthandler
import os
import sys
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from bmi.wrapper import BMIWrapper
from pydantic.class_validators import root_validator

from src.biota_models.coral.model.coral_model import Coral
from src.biota_models.vegetation.model.veg_model import Vegetation
from src.core.base_model import ExtraModel

faulthandler.enable()

WrapperVariable = Union[float, list, tuple, np.ndarray]


class Delft3D(ExtraModel, abc.ABC):
    """
    Implements the `HydrodynamicProtocol`.
    Coupling of coral_model to Delft3D using the BMI wrapper.
    """

    # Define model attributes.
    time_step: Optional[np.datetime64] = None
    model_wrapper: Optional[BMIWrapper] = None
    model_wrapper_dimr: Optional[BMIWrapper] = None
    d3d_home: Optional[Path] = None  # Delft3D binaries home directory.
    working_dir: Optional[Path] = None  # Model working directory.
    definition_file: Optional[Path] = None
    config_file: Optional[Path] = None
    dll_path: Optional[Path] = None

    update_interval: Optional[int] = None
    update_interval_storm: Optional[int] = None

    def __repr__(self):
        return "Delft3D()"

    @property
    @abstractmethod
    def space(self) -> Optional[int]:
        raise NotImplementedError

    def get_variable(self, variable: str) -> Optional[WrapperVariable]:
        """
        Get variable from the model wrapper.

        Args:
            variable (str): Variable to retrieve.

        Returns:
            Optional[WrapperVariable]: Value found.
        """
        return self.model_wrapper.get_var(variable)

    def set_variable(self, variable: str, value: Optional[WrapperVariable]):
        """
        Set variable to model wrapper.

        Args:
            variable (str): Variable to set.
            value (Optional[WrapperVariable]): Value to set.
        """
        self.model_wrapper.set_var(variable, value)

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
        if getattr(self.model_wrapper, obj) is None:
            msg = f"{obj} undefined (required for Delft3D coupling)"
            raise ValueError(msg)

    def reset_counters(self):
        """Reset properties for next model update."""
        sums = self.get_variable("is_sumvalsnd")
        sums.fill(0.0)
        self.set_variable("is_sumvalsnd", sums)

        maxs = self.get_variable("is_maxvalsnd")
        maxs.fill(0.0)
        self.set_variable("is_maxvalsnd", maxs)

    def set_morphology(self, coral: Coral):
        """Set morphological dimensions to Delft3D-model.

        :param coral: coral animal
        :type coral: Coral
        """
        self.set_variable("rnveg", coral.as_vegetation_density)
        self.set_variable("diaveg", coral.dc_rep)
        self.set_variable("stemheight", coral.hc)

    def set_vegetation(
        self, veg_species1: Vegetation, veg_species2: Optional[Vegetation], veg_species3: Optional[Vegetation]
    ):
        """Set vegetation dimensions to Delft3D-model.

        :param veg_species1: vegetation of a specific species
        :param veg_species2: vegetation of an optional second species
        :type veg_species1: Vegetation
        :type veg_species2: Vegetation
        """

        if not veg_species2 and not veg_species3:
            self.set_variable(
                "rnveg", veg_species1.veg_den
            )  # [1/m2] 3D plant density , 2D part is basis input (1/m2)
            self.set_variable(
                "diaveg", veg_species1.av_stemdia
            )  # [m] 3D plant diameter, 2D part is basis input (m)
            self.set_variable(
                "stemheight", veg_species1.av_height
            )  # [m] 2D plant heights (m)
        elif not veg_species3:  ## TODO TEST THIS!
            self.set_variable(
                "rnveg", (veg_species1.veg_den + veg_species2.veg_den)
            )  # [1/m2] 3D plant density , 2D part is basis input (1/m2)
            self.set_variable(
                "diaveg", (veg_species1.av_stemdia + veg_species2.av_stemdia)
            )  # [m] 3D plant diameter, 2D part is basis input (m)
            self.set_variable(
                "stemheight", (veg_species1.av_height + veg_species2.av_height)
            )  # [m] 2D plant heights (m)
        else:
            self.set_variable(
                "rnveg", (veg_species1.veg_den + veg_species2.veg_den + veg_species3.veg_den)
            )  # [1/m2] 3D plant density , 2D part is basis input (1/m2)
            self.set_variable(
                "diaveg", (veg_species1.av_stemdia + veg_species2.av_stemdia + veg_species3.av_stemdia)
            )  # [m] 3D plant diameter, 2D part is basis input (m)
            self.set_variable(
                "stemheight", (veg_species1.av_height + veg_species2.av_height + veg_species3.av_height)
            )  # [m] 2D plant heights (m)


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

    def get_hydromorphodynamics(self):
        """Get hydrodynamic results; max. values. And minimum in the future"""
        # TODO Add the minimum values when it is implemented in the model as a variable
        max_tau = self.get_variable("is_maxvalsnd")[range(self.space), 0]
        max_vel = self.get_variable("is_maxvalsnd")[range(self.space), 1]
        max_wl = self.get_variable("is_maxvalsnd")[range(self.space), 2]
        bed_level = self.get_variable("bl")

        return max_tau, max_wl, max_vel, bed_level

    def get_current_hydromorphodynamics(
        self, time_step
    ):  # only needed as long as we cannot get minval from the wrapper
        """Get hydrodynamic results; max. values. And minimum in the future"""
        self.time_step = time_step
        bed_level = self.get_variable("bl")[range(self.space)]
        # bed_level = np.delete(bed_level, np.where(bed_level <= -5))
        # cur_tau = self.get_variable('taus')
        # cur_vel = self.get_variable('u1')
        # cur1_vel = self.get_variable('ucx')
        # cur2_vel = self.get_variable('ucy')
        # cur_wl = self.get_variable('s1')
        dt_int = self.get_variable("is_dtint")
        cur_tau = (
            self.get_variable("is_sumvalsnd")[range(self.space), 0] / self.time_step
        )
        cur_vel = (
            self.get_variable("is_sumvalsnd")[range(self.space), 1] / self.time_step
        )
        cur_wl = (
            self.get_variable("is_sumvalsnd")[range(self.space), 2] / self.time_step
        )

        return cur_tau, cur_wl, cur_vel, bed_level

    @abstractmethod
    def configure_model_wrapper(self):
        """
        Configures the model wrapper with the specifics of its type.

        Raises:
            NotImplementedError: When the concrete class does not define its own implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_environment_variables(self) -> List[str]:
        """
        Gets the Python environment variables to include in a Delft3D model run.
        """
        raise NotImplementedError("Implement in concrete class")

    def _get_sys_environment_key(self) -> str:
        os_key = dict(win32="PATH", linux="LD_LIBRARY_PATH", darwin="DYLD_LIBRARY_PATH")
        env_key: str = os_key.get(sys.platform, None)
        if env_key is None:
            raise NotImplementedError(
                f"System {sys.platform} not supported for a Delft3D run."
            )
        return env_key

    def set_environment_variables(self):
        """
        Adds the required environment variables in to the systems path.
        Windows: PATH
        Linux: LD_LIBRARY_PATH
        Os (Darwin): DYLD_LIBRARY_PATH
        """
        env_variables = self.get_environment_variables()
        env_key = self._get_sys_environment_key()
        # Set variable
        path_var: str = os.environ[env_key]
        for env_var in env_variables:
            if str(env_var) not in path_var:
                path_var += f";{str(env_var)}"
        os.environ[env_key] = path_var

    def cleanup_environment_variables(self):
        """
        Remove unnecessary environment variables from system.
        """
        env_variables = self.get_environment_variables()
        env_key = self._get_sys_environment_key()
        # Set variable
        path_var: str = os.environ[env_key]
        for env_var in env_variables:
            if str(env_var) in path_var:
                path_var.replace(f";{str(env_var)}", "")
        os.environ[env_key] = path_var

    def initiate(self):
        """
        Creates a BMIWrapper and initializes it based on the given parameters for a FM Model.
        """
        self.set_environment_variables()
        self.configure_model_wrapper()
        if not self.model_wrapper_dimr:
            self.model_wrapper.initialize()
        else:
            self.model_wrapper_dimr.initialize()
    def update(self, coral, stormcat=0):
        """Update the Delft3D-model."""
        self.time_step = (
            self.update_interval_storm if stormcat > 0 else self.update_interval
        )
        self.reset_counters()
        self.model_wrapper.update(self.time_step)

        return (
            self.get_max_hydrodynamics()
            if stormcat > 0
            else self.get_mean_hydrodynamics()
        )

    ## TODO input timestep is in days! what is the unit here?
    def update_hydromorphodynamics(
        self,
        veg_species1: Vegetation,
        time_step: int,
        veg_species2: Optional[Vegetation] = None,
        veg_species3: Optional[Vegetation] = None,
    ):
        """Update the Delft3D-model.

        :param veg_species1: vegetation of a specific species
        :param time_step: time step of delft FM in seconds
        :param veg_species2: vegetation of an optional second species
        :type veg_species1: Vegetation
        :type time_step: int
        :type veg_species2: Vegetation

        """
        self.time_step = time_step
        self.reset_counters()
        self.set_vegetation(veg_species1, veg_species2, veg_species3)
        # if not veg_species2:
        #     self.set_vegetation(veg_species1)
        # else:
        #
        if not self.model_wrapper_dimr:
            self.model_wrapper.update(self.time_step)
        else:
            self.model_wrapper_dimr.update(self.time_step)
        return self.get_current_hydromorphodynamics(time_step=self.time_step)

    def finalise(self):
        """Finalize the working model."""
        self.model_wrapper.finalize()
        # self.cleanup_environment_variables()


class FlowFmModel(Delft3D):
    """
    Class implementing the `HydrodynamicProtocol` which makes use of a
    `BMIWrapper` to run its calculations.
    Based on a FlowFM model configuration.
    """

    _space: Optional[int] = None
    _water_depth: Optional[np.ndarray] = None
    _x_coordinates: Optional[np.array] = None
    _y_coordinates: Optional[np.array] = None

    @root_validator
    @classmethod
    def check_dll_path(cls, values: dict) -> dict:
        """
        Although not mandatory, we need to ensure at least a default value is given to the dll path.
        This default value is relative to the mandatory d3dhome attribute.

        Args:
            values (dict): Validated (and formatted) dictionary of values for a Delft3D object.

        Returns:
            dict: Validated dictionary with a `dll_path`.
        """
        dll_path_value = values.get("dll_path", None)
        if dll_path_value is None and values.get("d3d_home", None) is not None:
            values["dll_path"] = values["d3d_home"] / "dflowfm" / "bin" / "dflowfm.dll"
        return values

    @property
    def settings(self) -> str:
        incl = "DFlow-module"
        files = f"\n\tDFlow file         : {self.definition_file}"

        return (
            f"Coupling with Delft3D model (incl. {incl}) with the following settings:"
            f"\n\tDelft3D home dir.  : {self.d3d_home}"
            f"{files}"
        )

    @property
    def space(self) -> Optional[int]:
        """Number of non-boundary boxes; i.e. within-domain boxes."""
        if self.model_wrapper is None:
            return None
        self._space: Optional[np.ndarray] = (
            self.get_variable("ndxi") if self._space is None else self._space
        )
        return self._space.item()

    @property
    def water_depth(self) -> Optional[np.ndarray]:
        """Water depth."""
        if self.model_wrapper is None:
            return None
        if self.time_step is None:
            self.time_step = self.get_variable("is_dtint")
        if self._water_depth is None:
            return (
                self.get_variable("is_sumvalsnd")[range(self.space), 2] / self.time_step
            )
        else:
            return self._water_depth

    @property
    def x_coordinates(self) -> np.ndarray:
        """Center of gravity's x-coordinates as part of `space`."""
        if self.model_wrapper is None:
            return None
        self._x_coordinates = (
            self.get_variable("xzw")[range(self.space)]
            if self._x_coordinates is None
            else self._x_coordinates
        )
        return self._x_coordinates

    @property
    def y_coordinates(self) -> np.ndarray:
        """Center of gravity's y-coodinates as part of `space`."""
        if self.model_wrapper is None:
            return None
        self._y_coordinates = (
            self.get_variable("yzw")[range(self.space)]
            if self._y_coordinates is None
            else self._y_coordinates
        )
        return self._y_coordinates

    @property
    def xy_coordinates(self) -> np.ndarray:
        """The (x,y)-coordinates of the model domain,
        retrieved from hydrodynamic model; otherwise based on provided definition.

        :rtype: numpy.ndarray
        """
        if self.model_wrapper is None:
            return None
        return np.array(
            [
                [self.x_coordinates[i], self.y_coordinates[i]]
                for i in range(len(self.x_coordinates))
            ]
        )

    def get_environment_variables(self) -> List[str]:
        """Gets the Python environment variables required to run a FlowFM model."""
        return [
            self.d3d_home / "share" / "bin",
            self.d3d_home / "dflowfm" / "bin",
        ]

    def configure_model_wrapper(self):
        """
        Initilizes a BMIWrapper instance based on the given FlowFM parameters.
        Configures the model wrapper, it is recommended to set the environment variables beforehand.
        If the PATH variables does not work it is recommended copying all the contents from the share
        directory into the dimr bin dir.
        """
        self.model_wrapper = BMIWrapper(
            engine=self.dll_path.as_posix(), configfile=self.definition_file.as_posix()
        )


class DimrModel(Delft3D):
    """
    Class implementing the `HydrodynamicProtocol` which makes use of a
    `BMIWrapper` to run its calculations.
    Based on a DIMR model configuration.
    """
    _space: Optional[int] = None
    _water_depth: Optional[np.ndarray] = None
    _x_coordinates: Optional[np.array] = None
    _y_coordinates: Optional[np.array] = None

    @root_validator
    @classmethod
    def check_dll_path(cls, values: dict) -> dict:
        """
        Although not mandatory, we need to ensure at least a default value is given to the dll path.
        This default value is relative to the mandatory d3dhome attribute.

        Args:
            values (dict): Validated (and formatted) dictionary of values for a Delft3D object.

        Returns:
            dict: Validated dictionary with a `dll_path`.
        """
        dll_path_value = values.get("dll_path", None)
        if dll_path_value is None and values.get("d3d_home", None) is not None:
            values["dll_path"] = values["d3d_home"] / "dimr" / "bin" / "dimr_dll.dll"
        return values

    def get_environment_variables(self) -> List[str]:
        """
        Gets the Python environment variables required to run a Dimr model.
        """

        return [
            self.d3d_home / "share" / "bin",
            self.d3d_home / "dflowfm" / "bin",
            self.d3d_home / "dimr" / "bin",
            self.d3d_home / "dwaves" / "bin",
            self.d3d_home / "esmf" / "scripts",
            self.d3d_home / "swan" / "scripts",
        ]

    @property
    def settings(self) -> Path:
        incl = "DFlow- and DWaves-modules"
        files = (
            f"\n\tDFlow file         : {self.definition_file}"
            f"\n\tConfiguration file : {self.config_file}"
        )
        return (
            f"Coupling with Delft3D model (incl. {incl}) with the following settings:"
            f"\n\tDelft3D home dir.  : {self.d3d_home}"
            f"{files}"
        )

    # @property
    # def space(self) -> None:
    #     return None
    #
    # @property
    # def water_depth(self):
    #     return None
    #
    # @property
    # def x_coordinates(self):
    #     return None
    #
    # @property
    # def y_coordinates(self):
    #     return None
    #
    # @property
    # def xy_coordinates(self):
    #     return None

    @property
    def space(self) -> Optional[int]:
        """Number of non-boundary boxes; i.e. within-domain boxes."""
        if self.model_wrapper is None:
            return None
        self._space: Optional[np.ndarray] = (
            self.get_variable("ndxi") if self._space is None else self._space
        )
        return self._space.item()

    @property
    def water_depth(self) -> Optional[np.ndarray]:
        """Water depth."""
        if self.model_wrapper is None:
            return None
        if self.time_step is None:
            self.time_step = self.get_variable("is_dtint")
        if self._water_depth is None:
            return (
                self.get_variable("is_sumvalsnd")[range(self.space), 2] / self.time_step
            )
        else:
            return self._water_depth

    @property
    def x_coordinates(self) -> np.ndarray:
        """Center of gravity's x-coordinates as part of `space`."""
        if self.model_wrapper is None:
            return None
        self._x_coordinates = (
            self.get_variable("xzw")[range(self.space)]
            if self._x_coordinates is None
            else self._x_coordinates
        )
        return self._x_coordinates

    @property
    def y_coordinates(self) -> np.ndarray:
        """Center of gravity's y-coodinates as part of `space`."""
        if self.model_wrapper is None:
            return None
        self._y_coordinates = (
            self.get_variable("yzw")[range(self.space)]
            if self._y_coordinates is None
            else self._y_coordinates
        )
        return self._y_coordinates

    @property
    def xy_coordinates(self) -> np.ndarray:
        """The (x,y)-coordinates of the model domain,
        retrieved from hydrodynamic model; otherwise based on provided definition.

        :rtype: numpy.ndarray
        """
        if self.model_wrapper is None:
            return None
        return np.array(
            [
                [self.x_coordinates[i], self.y_coordinates[i]]
                for i in range(len(self.x_coordinates))
            ]
        )


    def configure_model_wrapper(self):
        """
        Initilizes a BMIWrapper instance based on the given DIMR parameters.
        It is recommended to set the environment variables beforehand.
        If the PATH variables does not work it is recommended copying all the contents from the share
        directory into the dimr bin dir.
        """
        ## Add corrects locations to environment variable PATH

        os.environ['PATH'] = os.path.join(self.d3d_home, 'dflowfm_with_shared', 'bin') \
        + ";" + os.path.join(self.d3d_home, 'esmf', 'scripts') \
        + ";" + os.path.join(self.d3d_home, 'swan', 'scripts')
        self.model_wrapper = BMIWrapper(os.path.join(self.d3d_home, 'dflowfm_with_shared', 'bin', 'dflowfm.dll'), configfile=self.definition_file)
        self.model_wrapper_dimr = BMIWrapper(
            engine=self.dll_path.as_posix(), configfile=self.config_file.as_posix()
        )
