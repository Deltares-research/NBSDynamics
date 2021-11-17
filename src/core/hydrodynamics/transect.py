import faulthandler
from pathlib import Path

import numpy as np
from src.core.base_model import ExtraModel
from typing import Optional

faulthandler.enable()


class Transect(ExtraModel):
    """
    Implements the `HydrodynamicProtocol`.
    Simple 1D depth transect with imposed currents and waves
    """

    working_dir: Optional[Path] = None
    definition_file: Optional[Path] = None
    config_file: Optional[Path] = None
    time_step: Optional[np.datetime64] = None

    x_coordinates: Optional[np.ndarray] = None
    y_coordinates: Optional[np.ndarray] = None
    water_depth: Optional[np.ndarray] = None
    outpoint: Optional[np.ndarray] = None  # Coordinates where his output is desired

    def __repr__(self):
        return "Transect()"

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
        if self.x_coordinates is None:
            return None
        return len(self.x_coordinates)

    @property
    def xy_coordinates(self) -> np.ndarray:
        """
        The (x,y)-coordinates of the model domain,
        retrieved from hydrodynamic model; otherwise based on provided definition.

        Returns:
            np.ndarray: The (x,y) coordinates.
        """
        if self.x_coordinates is None or self.y_coordinates is None:
            return None

        return np.array(
            [
                [self.x_coordinates[i], self.y_coordinates[i]]
                for i in range(len(self.x_coordinates))
            ]
        )

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
        """
        Initialize the working model.
        In this case, read the spatial configuration and the forcings
        from files. Set the computing environment.
        """
        csv: np.ndarray = np.genfromtxt(self.config_file, delimiter=",", skip_header=1)
        self.x_coordinates = csv[:, 0]
        self.y_coordinates = csv[:, 1]
        self.water_depth = csv[:, 2]
        self.outpoint = csv[:, 3] == 1

        forcings: np.ndarray = np.genfromtxt(
            self.definition_file, delimiter=",", skip_header=1
        )
        self.stormcat = forcings[:, 0]
        self.return_period = forcings[:, 1]
        self.wave_height = forcings[:, 2]
        self.wave_period = forcings[:, 3]
        self.wave_angle = forcings[:, 4]
        self.max_curr_vel = forcings[:, 5]

    def update(self, coral, stormcat=0):
        """
        Update the model, which is just knowing the waves

        Args:
            coral (Coral): Coral morphology to use.
            stormcat (int, optional): Storm category. Defaults to 0.

        Raises:
            ValueError: When stormcat not in [0,3] range.

        Returns:
            Tuple: Tuple containing calculated current velocity, wave velocity and wave period.
        """
        mean_current_vel = 0
        if stormcat in [0, 1, 2, 3]:
            Hs = self.wave_height[stormcat]
            T = self.wave_period[stormcat]
            max_current_vel = self.max_curr_vel[stormcat]
            h = self.water_depth
            wave_vel = (
                Hs
                / 4
                * np.sqrt(9.81 / h)
                * np.exp(-np.power((3.65 / T * np.sqrt(h / 9.81)), 2.1))
            )
        else:
            raise ValueError(f"stormcat = {stormcat}, must be either 0,1,2,3")
        if stormcat == 0:
            return mean_current_vel, wave_vel, T
        else:
            return max_current_vel, wave_vel, T

    def finalise(self):
        """Finalize the working model."""
        pass
