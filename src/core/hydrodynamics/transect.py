import faulthandler
import os
from pathlib import Path

import numpy as np

faulthandler.enable()


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
            f"\n\tTransect config file  : {self.config_file}"
            f"\n\tTransect forcings file : {self.definition_file}"
        )
        msg = (
            f"1D schematic cross-shore transect with forced hydrodynamics: "
            f"\n\tTransect model working dir.  : {self.working_dir}"
            f"{files}"
        )
        return msg

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
    def definition_file(self) -> Path:
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
    def config_file(self) -> Path:
        """Transect's config-file.

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

    def input_check(self):
        """Check if all requested content is provided"""

        self.input_check_definition("xy_coordinates")
        self.input_check_definition("water_depth")

        files = ("mdu", "config")
        [self.input_check_definition(file) for file in files]

    def input_check_definition(self, obj):
        """Check definition of critical object."""
        if getattr(self, obj) is None:
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
        pass

    def set_morphology(self, coral):
        """Set morphological dimensions to Delft3D-model.

        :param coral: coral animal
        :type coral: Coral
        """
        pass

    def initiate(self):
        """Initialize the working model.
        In this case, read the spatial configuration and the forcings
        from files. Set the computing environment.
        """
        csv = np.genfromtxt(self.config_file, delimiter=",", skip_header=1)
        self._x_coordinates = csv[:, 0]
        self._y_coordinates = csv[:, 1]
        self._water_depth = csv[:, 2]
        self._outpoint = csv[:, 3] == 1

        self.forcings = np.genfromtxt(
            self.definition_file, delimiter=",", skip_header=1
        )
        self.stormcat = self.forcings[:, 0]
        self.return_period = self.forcings[:, 1]
        self.wave_height = self.forcings[:, 2]
        self.wave_period = self.forcings[:, 3]
        self.wave_angle = self.forcings[:, 4]
        self.max_curr_vel = self.forcings[:, 5]

    def update(self, coral, stormcat=0):
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
