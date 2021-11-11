"""
coral_model - core

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""

import numpy as np

from src.core.constants import Constants
from src.core.utils import CoralOnly, DataReshape


class Coral:
    """Coral object, representing one coral type."""

    def __init__(self, constants: Constants, dc, hc, bc, tc, ac, species_constant=1):
        """
        :param dc: diameter coral plate [m]
        :param hc: coral height [m]
        :param bc: diameter coral base [m]
        :param tc: thickness coral plate [m]
        :param ac: axial distance corals [m]
        :param species_constant: species constant [-]

        :type dc: float, list, tuple, numpy.ndarray
        :type hc: float, list, tuple, numpy.ndarray
        :type bc: float, list, tuple, numpy.ndarray
        :type tc: float, list, tuple, numpy.ndarray
        :type ac: float, list, tuple, numpy.ndarray
        :type species_constant: float
        """
        self.constants = constants
        self.RESHAPE = DataReshape()
        self.dc = DataReshape.variable2array(dc)
        self.hc = DataReshape.variable2array(hc)
        self.bc = DataReshape.variable2array(bc)
        self.tc = DataReshape.variable2array(tc)
        self.ac = DataReshape.variable2array(ac)

        self.Csp = species_constant  # make into the constants list

        self._cover = None

        # initiate environmental working objects
        # > light micro-environment
        self.light = None
        self.light_bc = None  # Former self.Bc (code smell duplicated property name)
        # > flow micro-environment
        self.ucm = None
        self.um = None
        self.delta_t = None
        # > thermal micro-environment
        self.dTc = None
        self.temp = None
        # > photosynthesis
        self.photo_rate = None
        self.Tlo = None
        self.Thi = None
        # > population states
        self.pop_states = None
        self.p0 = None
        # np.array([
        #     self.cover, np.zeros(self.cover.shape), np.zeros(self.cover.shape), np.zeros(self.cover.shape),
        # ])
        # > calcification
        self.calc = None

    def __repr__(self):
        """Development representation."""
        return f"Morphology({self.dc}, {self.hc}, {self.bc}, {self.bc}, {self.ac})"

    def __str__(self):
        """Print representation."""
        return (
            f"Coral morphology with: dc = {self.dc} m; hc = {self.hc} ;"
            f"bc = {self.bc} m; tc = {self.tc} m; ac = {self.ac} m"
        )

    @property
    def dc_rep(self):
        """Representative coral diameter; weighted average of base and plate diameters."""
        return (self.bc * (self.hc - self.tc) + self.dc * self.tc) / self.hc

    @property
    def rf(self):
        """Form ratio: height-to-diameter ratio."""
        return self.hc / self.dc

    @property
    def rp(self):
        """Plate ratio: base-to-diameter ratio."""
        return self.bc / self.dc

    @property
    def rs(self):
        """Spacing ratio: diameter-to-axial distance ratio."""
        return self.dc / self.ac

    @property
    # changed the volume function - assigned the output coral_volume
    def volume(self):
        """Coral volume."""
        coral_volume = (
            0.25 * np.pi * ((self.hc - self.tc) * self.bc ** 2 + self.tc * self.dc ** 2)
        )
        return coral_volume

    @volume.setter  # what is the difference? And which volume does it call then?
    # also function update morphology does not update coral volume
    def volume(self, coral_volume):
        """
        :param coral_volume: coral volume [m3]
        :type coral_volume: float, int, list, tuple, np.ndarray
        """
        self.update_morphology(coral_volume, rf=self.rf, rp=self.rp, rs=self.rs)

    def update_morphology(self, coral_volume, rf, rp, rs):
        """Update the coral morphology based on updated coral volume and morphological ratios.

        :param coral_volume: coral volume [m3]
        :param rf: form ratio [-]
        :param rp: plate ratio [-]
        :param rs: spacing ratio [-]

        :type coral_volume: float, numpy.ndarray
        :type rf: float, numpy.ndarray
        :type rp: float, numpy.ndarray
        :type rs: float, numpy.ndarray
        """

        def vc2dc(coral_volume, rf, rp):
            """Coral volume to coral plate diameter."""
            dc = ((4.0 * coral_volume) / (np.pi * rf * rp * (1.0 + rp - rp ** 2))) ** (
                1.0 / 3.0
            )
            return dc

        def vc2hc(coral_volume, rf, rp):
            """Coral volume to coral height."""
            hc = (
                (4.0 * coral_volume * rf ** 2) / (np.pi * rp * (1.0 + rp - rp ** 2))
            ) ** (1.0 / 3.0)
            return hc

        def vc2bc(coral_volume, rf, rp):
            """Coral volume > diameter of the base."""
            bc = (
                (4.0 * coral_volume * rp ** 2) / (np.pi * rf * (1.0 + rp - rp ** 2))
            ) ** (1.0 / 3.0)
            return bc

        def vc2tc(coral_volume, rf, rp):
            """Coral volume > thickness of the plate."""
            tc = (
                (4.0 * coral_volume * rf ** 2 * rp ** 2)
                / (np.pi * (1.0 + rp - rp ** 2))
            ) ** (1.0 / 3.0)
            return tc

        def vc2ac(coral_volume, rf, rp, rs):
            """Coral volume > axial distance."""
            ac = (1.0 / rs) * (
                (4.0 * coral_volume) / (np.pi * rf * rp * (1.0 + rp - rp ** 2))
            ) ** (1.0 / 3.0)
            return ac

        # # update morphology
        self.dc = vc2dc(coral_volume, rf, rp)
        self.hc = vc2hc(coral_volume, rf, rp)
        self.bc = vc2bc(coral_volume, rf, rp)
        self.tc = vc2tc(coral_volume, rf, rp)
        self.ac = vc2ac(coral_volume, rf, rp, rs)

    @property
    def dc_matrix(self):
        """self.RESHAPEd coral plate diameter."""
        return self.RESHAPE.variable2matrix(self.dc, "space")

    @property
    def hc_matrix(self):
        """self.RESHAPEd coral height."""
        return self.RESHAPE.variable2matrix(self.hc, "space")

    @property
    def bc_matrix(self):
        """self.RESHAPEd coral base diameter."""
        return self.RESHAPE.variable2matrix(self.bc, "space")

    @property
    def tc_matrix(self):
        """self.RESHAPEd coral plate thickness."""
        return self.RESHAPE.variable2matrix(self.tc, "space")

    @property
    def ac_matrix(self):
        """self.RESHAPEd axial distance."""
        return self.RESHAPE.variable2matrix(self.ac, "space")

    @property
    def dc_rep_matrix(self):
        """self.RESHAPEd representative coral diameter."""
        return self.RESHAPE.variable2matrix(self.dc_rep, "space")

    @property
    def as_vegetation_density(self):
        """Translation from coral morphology to (vegetation) density."""

        def function(dc_rep, ac):
            return (2 * dc_rep) / (ac ** 2)

        return CoralOnly().in_space(
            coral=self, function=function, args=(self.dc_rep, self.ac)
        )

    @property
    def cover(self):
        """Carrying capacity."""
        if self._cover is None:
            cover = np.ones(np.array(self.volume).shape)
            cover[self.volume == 0.0] = 0.0  # 21.09 made 0. instead of just zero
            return cover

        return self._cover

    @cover.setter
    def cover(self, carrying_capacity):
        """
        :param carrying_capacity: carrying capacity [m2 m-2]
        :type carrying_capacity: float, list, tuple, numpy.ndarray
        """
        carrying_capacity = self.RESHAPE.variable2array(carrying_capacity)
        if not self.volume.shape == carrying_capacity.shape:
            raise ValueError(
                f"Shapes do not match: "
                f"{self.volume.shape} =/= {carrying_capacity.shape}"
            )

        if sum(self.volume[carrying_capacity == 0.0]) > 0.0:
            print(
                f"WARNING: Coral volume present where the carrying capacity is zero. This is unrealistic."
            )

        self._cover = carrying_capacity

    @property
    def living_cover(self):
        """Living coral cover based on population states."""
        if self.pop_states is not None:
            return self.pop_states.sum(axis=2)

    def initiate_spatial_morphology(self, cover=None):
        """Initiate the morphology based on the on set of morphological dimensions and the coral cover. This method
        contains a catch that it can only be used to initiate the morphology, and cannot overwrite existing spatial
        heterogeneous morphology definitions.

        :param cover: custom coral cover definition, defaults to None
        :type cover: None, numpy.ndarray
        """
        if cover is not None:
            cover = self.RESHAPE.variable2array(cover)
            if not cover.shape[0] == self.RESHAPE.space:
                msg = f"Spatial dimension of cover does not match: {cover.shape} =/= {self.RESHAPE.space}."
                raise ValueError(msg)
        else:
            cover = np.ones(self.RESHAPE.space)

        self.p0 = np.array(
            [
                cover,
                np.zeros(cover.shape),
                np.zeros(cover.shape),
                np.zeros(cover.shape),
            ]
        ).transpose()

        self.dc = cover * self.dc
        self.hc = cover * self.hc
        self.bc = cover * self.bc
        self.tc = cover * self.tc
        self.ac = cover * self.ac
