from typing import Optional

import numpy as np

from src.core.base_model import ExtraModel
from src.core.common.constants import Constants
from src.core.common.space_time import CoralOnly
from src.core.coral.coral_model import Coral


class Dislodgement(ExtraModel):
    """Dislodgement due to storm conditions."""

    constants: Constants = Constants()
    dmt: Optional[np.ndarray] = None
    csf: Optional[np.ndarray] = None
    survival: Optional[np.ndarray] = None

    def update(self, coral: Coral, survival_coefficient=1):
        """Update morphology due to storm damage.

        :param coral: coral animal
        :param survival_coefficient: percentage of partial survival, defualts to 1

        :type coral: Coral
        :type survival_coefficient: float, optional
        """
        # # partial dislodgement
        Dislodgement.partial_dislodgement(self, coral, survival_coefficient)
        # # update
        # ulation states
        for s in range(4):
            coral.p0[:, s] *= self.survival
        # morphology
        coral.update_coral_volume(coral.volume * self.survival)

    def partial_dislodgement(self, coral, survival_coefficient=1.0):
        """Percentage surviving storm event.

        :param coral: coral animal
        :param survival_coefficient: percentage of partial survival, defualts to 1

        :type coral: Coral
        :type survival_coefficient: float, optional
        """
        # TODO: Rewrite such that the distinction between an array or a float is well build in.
        try:
            self.survival = np.ones(coral.dc.shape)
        except TypeError:
            if Dislodgement.dislodgement_criterion(self, coral):
                self.survival = survival_coefficient * self.dmt / self.csf
            else:
                self.survival = 1.0
        else:
            dislodged = Dislodgement.dislodgement_criterion(self, coral)
            self.survival[dislodged] = survival_coefficient * (
                self.dmt[dislodged] / self.csf[dislodged]
            )

    def dislodgement_criterion(self, coral: Coral):
        """Dislodgement criterion. Returns boolean (array).

        :param coral: coral animal
        :type coral: Coral
        """
        self.dislodgement_mechanical_threshold(coral)
        self.colony_shape_factor(coral)
        return self.dmt <= self.csf

    def dislodgement_mechanical_threshold(self, coral: Coral):
        """
        Dislodgement Mechanical Threshold.

        Args:
            coral (Coral): Coral animal.
        """
        # # check input
        if not hasattr(coral.um, "__iter__"):
            coral.um = np.array([coral.um])
        if isinstance(coral.um, (list, tuple)):
            coral.um = np.array(coral.um)

        # # calculations
        self.dmt = 1e20 * np.ones(coral.um.shape)
        self.dmt[coral.um > 0] = self.dmt_formula(
            self.constants, coral.um[coral.um > 0]
        )

    @staticmethod
    def dmt_formula(constants, flow_velocity):
        """Determine the Dislodgement Mechanical Threshold.

        :param flow_velocity: depth-averaged flow velocity
        :type flow_velocity: float, numpy.ndarray
        """
        return constants.sigma_t / (constants.rho_w * constants.Cd * flow_velocity ** 2)

    def colony_shape_factor(self, coral: Coral):
        """
        Colony Shape Factor.

        Args:
            coral (Coral): Coral animal.
        """
        self.csf = CoralOnly().in_space(
            coral=coral,
            function=self.csf_formula,
            args=(coral.dc, coral.hc, coral.bc, coral.tc),
        )

    @staticmethod
    def csf_formula(dc, hc, bc, tc):
        """Determine the Colony Shape Factor.

        :param dc: diameter coral plate [m]
        :param hc: coral height [m]
        :param bc: diameter coral base [m]
        :param tc: thickness coral plate [m]

        :type dc: float, numpy.ndarray
        :type hc: float, numpy.ndarray
        :type bc: float, numpy.ndarray
        :type tc: float, numpy.ndarray

        :return: colony shape factor
        :rtype: float, numpy.ndarray
        """
        # arms of moment
        arm_top = hc - 0.5 * tc
        arm_bottom = 0.5 * (hc - tc)
        # area of moment
        area_top = dc * tc
        area_bottom = bc * (hc - tc)
        # integral
        integral = arm_top * area_top + arm_bottom * area_bottom
        # colony shape factor
        return 16.0 / (np.pi * bc ** 3) * integral
