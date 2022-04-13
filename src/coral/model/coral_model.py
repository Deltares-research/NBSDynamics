from typing import Dict, Optional, Union

import numpy as np
from pydantic import validator

from src.coral.model.coral_constants import CoralConstants
from src.coral.model.coral_only import CoralOnly
from src.core import RESHAPE
from src.core.base_model import ExtraModel
from src.core.biota.biota_model import Biota
from src.core.common.space_time import DataReshape

CoralAttribute = Union[float, list, tuple, np.ndarray]


class Coral(Biota):

    """
    Implements the `CoralProtocol`.
    Coral object, representing one coral type.
    """

    constants: CoralConstants = CoralConstants()
    dc: CoralAttribute  # diameter coral plate [m]
    hc: CoralAttribute  # coral height [m]
    bc: CoralAttribute  # diameter coral base [m]
    tc: CoralAttribute  # thickness coral plate [m]
    ac: CoralAttribute  # axial distance corals [m]
    Csp: Optional[float] = 1  # species constant [-]

    # other attributes.
    _cover: Optional[CoralAttribute] = None
    # light micro-environment
    light: Optional[CoralAttribute] = None
    light_bc: Optional[CoralAttribute] = None
    # flow micro environment
    ucm: Optional[CoralAttribute] = None
    um: Optional[CoralAttribute] = None
    delta_t: Optional[CoralAttribute] = None
    # thermal micro-environment
    dTc: Optional[CoralAttribute] = None
    temp: Optional[CoralAttribute] = None
    # photosynthesis
    photo_rate: Optional[CoralAttribute] = None
    Tlo: Optional[CoralAttribute] = None
    Thi: Optional[CoralAttribute] = None
    # population states
    pop_states: Optional[CoralAttribute] = None
    p0: Optional[CoralAttribute] = None
    # calcification
    calc: Optional[CoralAttribute] = None

    @validator("dc", "hc", "bc", "tc", "ac")
    @classmethod
    def validate_coral_attribute(
        cls, value: Optional[CoralAttribute]
    ) -> Optional[np.ndarray]:
        if value is None:
            return value
        return DataReshape.variable2array(value)

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

    @property
    def dc_matrix(self):
        """self.RESHAPEd coral plate diameter."""
        return RESHAPE().variable2matrix(self.dc, "space")

    @property
    def hc_matrix(self):
        """self.RESHAPEd coral height."""
        return RESHAPE().variable2matrix(self.hc, "space")

    @property
    def bc_matrix(self):
        """self.RESHAPEd coral base diameter."""
        return RESHAPE().variable2matrix(self.bc, "space")

    @property
    def tc_matrix(self):
        """self.RESHAPEd coral plate thickness."""
        return RESHAPE().variable2matrix(self.tc, "space")

    @property
    def ac_matrix(self):
        """self.RESHAPEd axial distance."""
        return RESHAPE().variable2matrix(self.ac, "space")

    @property
    def dc_rep_matrix(self):
        """self.RESHAPEd representative coral diameter."""
        return RESHAPE().variable2matrix(self.dc_rep, "space")

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

    @property
    def living_cover(self):
        """Living coral cover based on population states."""
        if self.pop_states is not None:
            return self.pop_states.sum(axis=2)

    def update_coral_volume(self, coral_volume: CoralAttribute):
        """
        Updates the coral morphology based on the given coral volume.

        Args:
            coral_volume (CoralAttribute): New coral volume.
        """
        # TODO what is the difference? And which volume does it call then?
        # TODO also function update morphology does not update coral volume
        self.update_coral_morphology(
            coral_volume, dict(rf=self.rf, rp=self.rp, rs=self.rs)
        )

    def update_cover(self, carrying_capacity: CoralAttribute):
        """
        Update cover value based on given parameters.

        Args:
            carrying_capacity (CoralAttribute): Carrying capacity [m2 m-2].
        """
        carrying_capacity = RESHAPE().variable2array(carrying_capacity)
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

    def initiate_coral_morphology(self, cover: Optional[np.ndarray] = None):
        """
        Initiate the morphology based on the on set of morphological dimensions and the coral cover. This method
        contains a catch that it can only be used to initiate the morphology, and cannot overwrite existing spatial
        heterogeneous morphology definitions.

        Args:
            cover (Optional[np.ndarray]): Custom coral definition.
        """
        _reshape = RESHAPE()
        if cover is not None:
            cover = _reshape.variable2array(cover)
            if not cover.shape[0] == _reshape.space:
                msg = f"Spatial dimension of cover does not match: {cover.shape} =/= {_reshape.space}."
                raise ValueError(msg)
        else:
            cover = np.ones(_reshape.space)

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

    def update_coral_morphology(
        self,
        coral_volume: Union[float, np.ndarray],
        morphology_ratios: Dict[str, Union[float, np.ndarray]],
    ):
        """
        Update the coral morphology based on updated coral volume and morphology ratios.

        Args:
            coral_volume (Union[float, np.ndarray]): Coral volume
            morphology_ratios (Dict[str, Union[float, np.ndarray]]): Morphology ratios (rf, rp, rs, ..)
        """
        rf = morphology_ratios["rf"]
        rp = morphology_ratios["rp"]
        rs = morphology_ratios["rs"]

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
