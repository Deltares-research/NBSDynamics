import faulthandler
import os
from pathlib import Path

import numpy as np
from bmi.wrapper import BMIWrapper

faulthandler.enable()


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
        if self.config_file:
            incl = f"DFlow- and DWaves-modules"
            files = (
                f"\n\tDFlow file         : {self.definition_file}"
                f"\n\tConfiguration file : {self.config_file}"
            )
        else:
            incl = f"DFlow-module"
            files = f"\n\tDFlow file         : {self.definition_file}"

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
        if not isinstance(folder, Path):
            self._home = Path(folder)
            return
        self._home = folder

    @property
    def working_dir(self) -> Path:
        """Model working directory."""
        return self._working_dir

    @working_dir.setter
    def working_dir(self, folder):
        """
        :param folder: working directory
        :type folder:  str
        """
        if not isinstance(folder, Path):
            self._working_dir = Path(folder)
            return
        self._working_dir = folder

    @property
    def dflow_dir(self) -> Path:
        """Directory to DFlow-ddl."""
        self._dflow_dir = self.d3d_home / "dflowfm" / "bin" / "dflowfm"
        return self._dflow_dir

    @property
    def dimr_dir(self) -> Path:
        """Directory to DIMR-dll."""
        self._dimr_dir = self.d3d_home / "dimr" / "bin" / "dimr_dll"
        return self._dimr_dir

    @property
    def definition_file(self):
        """Delft3D's MDU-file.

        :rtype: str
        """
        return self._mdu

    @definition_file.setter
    def definition_file(self, file_dir):
        """
        :param file_dir: file directory of MDU-file
        :type file_dir: str
        """
        self._mdu = self.working_dir / file_dir

    @property
    def config_file(self):
        """Delft3D's config-file.

        :rtype: str
        """
        return self._config

    @config_file.setter
    def config_file(self, file_dir):
        """
        :param file_dir: file directory of config-file
        :type file_dir: str, list, tuple
        """
        self._config = self.working_dir / file_dir

    @property
    def model(self):
        """Main model-object."""
        return self.model_dimr if self.config_file else self.model_fm

    @property
    def model_fm(self):
        """Deflt3D-FM model-object."""
        if self._model_fm is None:
            raise ValueError("Model FM has not been defined.")
        return self._model_fm

    @property
    def model_dimr(self):
        """Delft3D DIMR model-object."""
        if self._model_dimr is None:
            raise ValueError("Model dimr has not been defined.")
        return self._model_dimr

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
            self.time_step = self.get_variable("is_dtint")
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
        self._model_fm = BMIWrapper(
            engine=self.dflow_dir, configfile=self.definition_file
        )
        if self.config_file:
            self._model_dimr = BMIWrapper(
                engine=self.dimr_dir, configfile=self.config_file
            )
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
