from typing import Optional
import numpy as np
from pathlib import Path

from src.core.coral_model import Coral


class Reef0D:
    """Implements the 'HydrodynamicProtocol'."""

    @property
    def settings(self):
        """Print settings of Reef0D-model."""
        msg = f"Not yet implemented."
        return msg

    @property
    def working_dir(self) -> Optional[Path]:
        return None

    @property
    def config_file(self) -> Optional[Path]:
        return None

    @property
    def definition_file(self) -> Optional[Path]:
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
        if not self.x_coordinates or self.y_coordinates:
            raise ValueError(
                "XY Coordinates require both 'x_coorinates' and 'y_coordinates' to be defined."
            )
        return np.array(
            [
                [self.x_coordinates[i], self.y_coordinates[i]]
                for i in range(len(self.x_coordinates))
            ]
        )

    @property
    def space(self):
        return len(self.xy_coordinates)

    def initiate(self):
        """Initiate hydrodynamic model."""
        raise NotImplementedError

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
        raise NotImplementedError
