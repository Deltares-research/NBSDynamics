from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import fsolve

from src.core.coral_model import Coral


class Reef1D:
    """
    Implements the `HydrodynamicProtocol`.
    Simplified one-dimensional hydrodynamic model over a (coral) reef."""

    # TODO: Complete the one-dimensional hydrodynamic model
    def __init__(self):
        """Internal 1D hydrodynamic model for order-of-magnitude calculations on the hydrodynamic conditions on a coral
        reef, where both flow and waves are included.
        """
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
    def working_dir(self) -> Optional[Path]:
        return None

    @property
    def config_file(self) -> Optional[Path]:
        return None

    @property
    def definition_file(self) -> Optional[Path]:
        return None

    @property
    def settings(self):
        """Print settings of Reef1D-model."""
        bath_value = None
        min_max_bath = None
        space_dx = None

        if self.bath is not None:
            bath_value = type(self.bath).__name__
            min_max_bath = f"{min(self.bath)}-{max(self.bath)}"
            if self.dx is not None:
                space_dx = self.space * self.dx

        msg = (
            f"One-dimensional simple hydrodynamic model to simulate the "
            f"hydrodynamics on a (coral) reef with the following settings:"
            f"\n\tBathymetric cross-shore data : {bath_value}"
            f"\n\t\trange [m]  : {min_max_bath}"
            f"\n\t\tlength [m] : {space_dx}"
            f"\n\tSignificant wave height [m]  : {self.Hs}"
            f"\n\tPeak wave period [s]         : {self.Tp}"
        )
        return msg

    @property
    def space(self):
        if self.bath is None:
            return None
        return len(self.bath)

    @property
    def water_depth(self):
        return None

    @property
    def x_coordinates(self):
        if self.space is None or self.dx is None:
            return None
        return np.arange(0, self.space, self.dx)

    @property
    def y_coordinates(self):
        return np.array([0])

    @property
    def xy_coordinates(self):
        if self.x_coordinates is None:
            return None
        return np.array(
            [
                [self.x_coordinates[i], self.y_coordinates[0]]
                for i in range(len(self.x_coordinates))
            ]
        )

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
