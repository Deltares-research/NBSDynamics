import abc
import faulthandler
import os
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
from bmi.wrapper import BMIWrapper
from pydantic import Extra

from src.core.base_model import BaseModel

faulthandler.enable()

WrapperVariable = Union[float, list, tuple, np.ndarray]


class Delft3D(BaseModel, abc.ABC):
    """
    Implements the `HydrodynamicProtocol`.
    Coupling of coral_model to Delft3D using the BMI wrapper.
    """

    class Config:
        """
        Allow this model to have extra fields defined during runtime.
        """

        extra = Extra.allow

    # Define model attributes.
    time_step: Optional[np.datetime64]
    model_wrapper: Optional[BMIWrapper]
    d3d_home: Optional[Path]  # Delft3D binaries home directory.
    working_dir: Optional[Path]  # Model working directory.
    definition_file: Optional[Path] = None
    config_file: Optional[Path] = None

    @property
    @abstractmethod
    def dll_dir(self) -> Path:
        raise NotImplementedError

    def __repr__(self):
        return "Delft3D()"

    def environment(self):
        """Set Python environment to include Delft3D-code."""
        dirs = [
            self.d3d_home / "share" / "bin",
            self.d3d_home / "dflowfm" / "bin",
        ]
        if self.config_file:
            dirs.extend(
                [
                    self.d3d_home / "dimr" / "bin",
                    self.d3d_home / "dwaves" / "bin",
                    self.d3d_home / "esmf" / "scripts",
                    self.d3d_home / "swan" / "scripts",
                ]
            )

        env = ";".join(map(str, dirs))
        os.environ["PATH"] = env

        print('\nEnvironment "PATH":')
        [print(f"\t{path}") for path in dirs]

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

    def set_update_intervals(self, default: int, storm: Optional[int] = None):
        """
        Set update intervals

        Args:
            default (int): Default value to update.
            storm (Optional[int], optional): Default value if none given. Defaults to None.
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

    @abstractmethod
    def configure_model_wrapper(self):
        """
        Configures the model wrapper with the specifics of its type.

        Raises:
            NotImplementedError: When the concrete class does not define its own implementation.
        """
        raise NotImplementedError

    def initiate(self):
        """
        Creates a BMIWrapper and initializes it based on the given parameters for a FM Model.
        """
        self.environment()
        self.configure_model_wrapper()
        self.model_wrapper.initialize()

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

    def finalise(self):
        """Finalize the working model."""
        self.model_wrapper.finalize()


class FlowFmModel(Delft3D):
    """
    Class implementing the `HydrodynamicProtocol` which makes use of a
    `BMIWrapper` to run its calculations.
    Based on a FlowFM model configuration.
    """

    _space: Optional[int] = None
    _water_depth: Optional[np.ndarray] = None
    _x_coordinates: Optional[np.array]
    _y_coordinates: Optional[np.array]

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
    def dll_dir(self) -> Path:
        return self.d3d_home / "dflowfm" / "bin" / "dflowfm"

    @property
    def space(self) -> int:
        """Number of non-boundary boxes; i.e. within-domain boxes."""
        if self.model_wrapper is None:
            return None
        self._space: Optional[int] = (
            self.get_variable("ndxi") if self._space is None else self._space
        )
        return self._space.item()

    @property
    def water_depth(self) -> np.ndarray:
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
        self.model_wrapper = BMIWrapper(
            engine=self.dll_dir.as_posix(), configfile=self.definition_file.as_posix()
        )


class DimrModel(Delft3D):
    """
    Class implementing the `HydrodynamicProtocol` which makes use of a
    `BMIWrapper` to run its calculations.
    Based on a DIMR model configuration.
    """

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

    @property
    def dll_dir(self) -> Path:
        return self.d3d_home / "dimr" / "bin" / "dimr_dll"

    @property
    def space(self) -> None:
        return None

    @property
    def water_depth(self):
        return None

    @property
    def x_coordinates(self):
        return None

    @property
    def y_coordinates(self):
        return None

    @property
    def xy_coordinates(self):
        return None

    def configure_model_wrapper(self):
        """
        Initilizes a BMIWrapper instance based on the given DIMR parameters.
        """
        self.model_wrapper = BMIWrapper(
            engine=self.dll_dir.as_posix(), configfile=self.config_file.as_posix()
        )
