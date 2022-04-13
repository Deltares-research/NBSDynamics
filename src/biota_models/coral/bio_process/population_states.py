from typing import Optional

import numpy as np

from src.biota_models.coral.model.coral_constants import CoralConstants
from src.biota_models.coral.model.coral_model import Coral
from src.core import RESHAPE


class PopulationStates:
    """Bleaching response following the population dynamics."""

    # TODO: Check this class; incl. writing tests

    def __init__(
        self, constants: CoralConstants = CoralConstants(), dt: Optional[float] = 1
    ):
        """Population dynamics.

        :param dt: time step [yrs], defaults to one
        :type dt: float, optional
        """
        self.dt = dt
        self.constants = constants

    def pop_states_t(self, coral: Coral):
        """Population dynamics over time.

        :param coral: coral animal
        :type coral: Coral
        """
        _reshape = RESHAPE()
        coral.pop_states = np.zeros((*_reshape.spacetime, 4))
        for n in range(_reshape.time):
            photosynthesis = np.zeros(_reshape.space)
            photosynthesis[coral.cover > 0.0] = coral.photo_rate[
                coral.cover > 0.0, n
            ]  # 21_09 have changed coral.cover>0 to .0.
            coral.pop_states[:, n, :] = self.pop_states_xy(coral, photosynthesis)
            coral.p0[coral.cover > 0.0, :] = coral.pop_states[coral.cover > 0.0, n, :]

    def pop_states_xy(self, coral: Coral, ps):
        """Population dynamics over space.

        :param coral: coral animal
        :param ps: photosynthetic rate

        :type coral: Coral
        :type ps: numpy.ndarray
        """
        p = np.zeros((RESHAPE().space, 4))
        # # calculations
        # growing conditions
        # > bleached pop.      # ps>0. here represents ps>tsh that is the value of the bleaching treshold light and 1. where 1.0 is a number, not column reference
        p[ps > 0.0, 3] = coral.p0[ps > 0.0, 3] / (
            1
            + self.dt
            * (
                8.0 * self.constants.r_recovery * ps[ps > 0.0] / coral.Csp
                + self.constants.r_mortality * coral.Csp
            )
        )
        # > pale pop.
        p[ps > 0.0, 2] = (
            coral.p0[ps > 0.0, 2]
            + (8.0 * self.dt * self.constants.r_recovery * ps[ps > 0.0] / coral.Csp)
            * p[ps > 0.0, 3]
        ) / (1.0 + self.dt * self.constants.r_recovery * ps[ps > 0.0] * coral.Csp)
        # > recovering pop.
        p[ps > 0.0, 1] = (
            coral.p0[ps > 0.0, 1]
            + self.dt
            * self.constants.r_recovery
            * ps[ps > 0.0]
            * coral.Csp
            * p[ps > 0.0, 2]
        ) / (1.0 + 0.5 * self.dt * self.constants.r_recovery * ps[ps > 0.0] * coral.Csp)
        # > healthy pop.
        a = (
            self.dt
            * self.constants.r_growth
            * ps[ps > 0.0]
            * coral.Csp
            / coral.cover[ps > 0.0]
        )
        b = 1.0 - self.dt * self.constants.r_growth * ps[ps > 0.0] * coral.Csp * (
            1.0 - p[ps > 0.0, 1:].sum(axis=1) / coral.cover[ps > 0.0]
        )
        c = -(
            coral.p0[ps > 0.0, 0]
            + 0.5
            * self.dt
            * self.constants.r_recovery
            * ps[ps > 0.0]
            * coral.Csp
            * p[ps > 0.0, 1]
        )
        p[ps > 0.0, 0] = (-b + np.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)

        # bleaching conditions
        # > healthy pop.
        p[ps <= 0.0, 0] = coral.p0[ps <= 0.0, 0] / (
            1.0 - self.dt * self.constants.r_bleaching * ps[ps <= 0.0] * coral.Csp
        )
        # > recovering pop.
        p[ps <= 0.0, 1] = coral.p0[ps <= 0.0, 1] / (
            1.0 - self.dt * self.constants.r_bleaching * ps[ps <= 0.0] * coral.Csp
        )
        # > pale pop.
        p[ps <= 0.0, 2] = (
            coral.p0[ps <= 0.0, 2]
            - self.dt
            * self.constants.r_bleaching
            * ps[ps <= 0.0]
            * coral.Csp
            * (p[ps <= 0.0, 0] + p[ps <= 0.0, 1])
        ) / (
            1.0 - 0.5 * self.dt * self.constants.r_bleaching * ps[ps <= 0.0] * coral.Csp
        )
        # > bleached pop.
        p[ps <= 0.0, 3] = (
            coral.p0[ps <= 0.0, 3]
            - 0.5
            * self.dt
            * self.constants.r_bleaching
            * ps[ps <= 0]
            * coral.Csp
            * p[ps <= 0.0, 2]
        ) / (
            1.0
            - 0.25 * self.dt * self.constants.r_bleaching * ps[ps <= 0.0] * coral.Csp
        )

        # # check on carrying capacity
        if any(p.sum(axis=1) > 1.0001 * coral.cover):
            slot_1 = np.arange(len(coral.cover))[p.sum(axis=1) > 1.0001 * coral.cover]
            slot_2 = p[p.sum(axis=1) > 1.0001 * coral.cover]
            slot_3 = coral.cover[p.sum(axis=1) > 1.0001 * coral.cover]
            print(
                f"WARNING: Total population than carrying capacity at {slot_1}. "
                f"\n\tPT = {slot_2}; K = {slot_3}"
            )

        # # output
        return p
