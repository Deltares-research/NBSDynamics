from pathlib import Path
from typing import Optional

import numpy as np
from src.core.base_model import BaseModel

from src.core.coral.coral_model import Coral


class Reef0D(BaseModel):
    """Implements the `HydrodynamicProtocol`."""

    working_dir: Optional[Path] = None
    definition_file: Optional[Path] = None
    config_file: Optional[Path] = None
    x_coordinates: Optional[np.ndarray] = None
    y_coordinates: Optional[np.ndarray] = None
    water_depth: Optional[np.ndarray] = None

    @property
    def settings(self):
        """Print settings of Reef0D-model."""
        return "Not yet implemented."

    @property
    def xy_coordinates(self):
        if self.x_coordinates is None or self.y_coordinates is None:
            return None
        return np.array(
            [
                [self.x_coordinates[i], self.y_coordinates[i]]
                for i in range(len(self.x_coordinates))
            ]
        )

    @property
    def space(self):
        if self.xy_coordinates is None:
            return None
        return len(self.xy_coordinates)

    def initiate(self):
        """Initiate hydrodynamic model."""
        raise NotImplementedError

    def update(self, coral: Coral, storm=False) -> tuple:
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
