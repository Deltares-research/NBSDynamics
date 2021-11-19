from typing import Optional, Union

import numpy as np

from src.core import RESHAPE
from src.core.common.constants import Constants
from src.core.common.space_time import DataReshape
from src.core.coral.coral_only import CoralOnly

LightVariable = Union[float, list, tuple, np.ndarray]


class Light:
    """Light micro-environment."""

    def __init__(
        self,
        light_in: LightVariable,
        lac: LightVariable,
        depth: LightVariable,
        constants: Constants = Constants(),
    ):
        """
        Light micro-environment.

        Args:
            light_in (LightVariable): Incoming light-intensity at the water-air interface [u mol photons m-2 s-1]
            lac (LightVariable): light-attenuation coefficient [m-1]
            depth (LightVariable): water depth [m]
            constants (Constants): List of constants to be used for this object.
        """
        _reshape = RESHAPE()
        self.I0 = _reshape.variable2matrix(light_in, "time")
        self.Kd = _reshape.variable2matrix(lac, "time")
        self.h = _reshape.variable2matrix(depth, "space")
        self.constants = constants

    def rep_light(self, coral):
        """Representative light-intensity.

        :param coral: coral animal
        :type coral: Coral
        """
        base_section = self.base_light(coral)
        # # light catchment per coral section
        # top of plate
        top = (
            0.25
            * np.pi
            * coral.dc_matrix ** 2
            * self.I0
            * np.exp(-self.Kd * (self.h - coral.hc_matrix))
        )
        # side of plate
        side = (
            (np.pi * coral.dc_matrix * self.I0)
            / self.Kd
            * (
                np.exp(-self.Kd * (self.h - coral.hc_matrix))
                - np.exp(-self.Kd * (self.h - coral.hc_matrix + coral.tc_matrix))
            )
            * self.side_correction(coral)
        )
        # side of base
        base = (
            (np.pi * coral.bc_matrix * self.I0)
            / self.Kd
            * (np.exp(-self.Kd * (self.h - base_section)) - np.exp(-self.Kd * self.h))
            * self.side_correction(coral)
        )
        # total
        total = top + side + base

        # # biomass-averaged
        self.biomass(coral)

        def averaged_light(total_light, biomass):
            """Averaged light-intensity."""
            return total_light / biomass

        coral.light = CoralOnly().in_spacetime(
            coral=coral,
            function=averaged_light,
            args=(total, coral.light_bc),
            no_cover_value=self.I0 * np.exp(-self.Kd * self.h),
        )

    def biomass(self, coral):
        """Coral biomass; as surface.

        :param coral: coral animal
        :type coral: Coral
        """
        base_section = self.base_light(coral)
        coral.light_bc = np.pi * (
            0.25 * coral.dc_matrix ** 2
            + coral.dc_matrix * coral.tc_matrix
            + coral.bc_matrix * base_section
        )

    def base_light(self, coral):
        """Section of coral base receiving light.

        :param coral: coral animal
        :type coral: Coral
        """
        # # spreading of light
        theta = self.light_spreading(coral)

        # # coral base section
        base_section = (
            coral.hc_matrix
            - coral.tc_matrix
            - ((coral.dc_matrix - coral.bc_matrix) / (2.0 * np.tan(0.5 * theta)))
        )
        # no negative lengths
        base_section[base_section < 0] = 0

        return base_section

    def light_spreading(self, coral):
        """Spreading of light as function of depth.

        :param coral: coral animal
        :type coral: Coral
        """
        return self.constants.theta_max * np.exp(
            -self.Kd * (self.h - coral.hc_matrix + coral.tc_matrix)
        )

    def side_correction(self, coral):
        """Correction of the light-intensity on the sides of the coral object.

        :param coral: coral animal
        :type coral: Coral
        """
        # # spreading of light
        theta = self.light_spreading(coral)

        # # correction factor
        return np.sin(0.5 * theta)
