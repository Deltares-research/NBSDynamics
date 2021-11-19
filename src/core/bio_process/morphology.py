import numpy as np

from src.core import RESHAPE
from src.core.common.constants import Constants
from src.core.coral.coral_model import Coral


class Morphology:
    """Morphological development."""

    __rf_optimal = None
    __rp_optimal = None
    __rs_optimal = None

    def __init__(
        self, calc_sum, light_in, dt_year=1, constants: Constants = Constants()
    ):
        """
        Morphological development.

        :param calc_sum: accumulation of calcification of :param dt_year: years [kg m-2 yr-1]
        :param light_in: incoming light-intensity at water-air interface [umol photons m-2 s-1]
        :param dt_year: update interval [yr], defaults to 1

        :type calc_sum: float, int, list, tuple, numpy.ndarray
        :type light_in: float, int, list, tuple, numpy.ndarray
        :type dt_year: float, int
        """
        _reshape = RESHAPE()
        try:
            _ = len(calc_sum[0])
        except TypeError:
            self.calc_sum = calc_sum
        else:
            self.calc_sum = _reshape.matrix2array(calc_sum, "space", "sum")
        self.dt_year = dt_year

        self.I0 = _reshape.variable2matrix(light_in, "time")
        self.vol_increase = 0

        self.constants = constants

    @staticmethod
    def __coral_object_checker(coral):
        """Check the suitability of the coral-object for the morphological development.

        :param coral: coral animal
        :type coral: Coral
        """
        # coral must be of type Coral
        if not isinstance(coral, Coral):
            msg = f"The optimal ratios are set using the Coral-object, {type(coral)} is given."
            raise TypeError(msg)

        # coral must have light and flow condition attributes
        if not hasattr(coral, "light") and not hasattr(coral, "ucm"):
            msg = (
                f"The optimal ratios are determined based on the coral's light and flow conditions; "
                f"none are provided."
            )
            raise AttributeError(msg)

    @property
    def rf_optimal(self):
        """Optimal form ratio; height-to-plate diameter.

        :rtype: float, numpy.ndarray
        """
        return self.__rf_optimal

    @rf_optimal.setter
    def rf_optimal(self, coral):
        """
        :param coral: coral animal
        :type coral: Coral
        """
        self.__coral_object_checker(coral)

        rf = self.constants.rf

        # rf = (
        #     self.constants.prop_form
        #     * (coral.light.mean(axis=1) / self.I0.mean(axis=1))
        #     * (self.constants.u0 / 1e-6)
        # )
        # rf[coral.ucm > 0] = (
        #     self.constants.prop_form
        #     * (
        #         coral.light.mean(axis=1)[coral.ucm > 0]
        #         / self.I0.mean(axis=1)[coral.ucm > 0]
        #     )
        #     * (self.constants.u0 / coral.ucm[coral.ucm > 0])
        # )
        self.__rf_optimal = rf

    @property
    def rp_optimal(self):
        """Optimal plate ratio; base diameter-to-plate diameter.

        :rtype: float, numpy.ndarray
        """
        return self.__rp_optimal

    @rp_optimal.setter
    def rp_optimal(self, coral):
        """
        :param coral: coral animal
        :type coral: Coral
        """
        self.__coral_object_checker(coral)
        self.__rp_optimal = self.constants.rp

        # self.__rp_optimal = self.constants.prop_plate * (
        #     1.0
        #     + np.tanh(
        #         self.constants.prop_plate_flow
        #         * (coral.ucm - self.constants.u0)
        #         / self.constants.u0
        #     )
        # )

    @property
    def rs_optimal(self):
        """Optimal spacing ratio; plate diameter-to-axial distance.

        :rtype: float, numpy.ndarray
        """
        return self.__rs_optimal

    @rs_optimal.setter
    def rs_optimal(self, coral):
        """
        :param coral: coral animal
        :type coral: Coral
        """
        self.__coral_object_checker(coral)

        # self.__rs_optimal = 0.5 / np.sqrt(2.0) * 0.25

        self.__rs_optimal = (
            self.constants.prop_space
            * (
                1.0
                - np.tanh(
                    self.constants.prop_space_light
                    * coral.light.mean(axis=1)
                    / self.I0.mean(axis=1)
                )
            )
            * (
                1.0
                + np.tanh(
                    self.constants.prop_space_flow
                    * (coral.ucm - self.constants.u0)
                    / self.constants.u0
                )
            )
        )

    def delta_volume(self, coral):
        """
        :param coral: coral object
        :type coral: Coral
        """
        self.vol_increase = (
            0.5
            * coral.ac ** 2
            * self.calc_sum
            * self.dt_year
            / self.constants.rho_c
            * coral.light_bc.mean(axis=1)
        )
        return self.vol_increase

    def ratio_update(self, coral, ratio):
        """
        :param coral: coral object
        :param ratio: morphological ratio to update

        :type coral: Coral
        :type ratio: str
        """

        # partial differential equation - mass balance
        def mass_balance(r_old, r_opt):
            """Mass balance."""
            return (coral.volume * r_old + self.vol_increase * r_opt) / (
                coral.volume + self.vol_increase
            )

        # input check
        ratios = ("rf", "rp", "rs")
        if ratio not in ratios:
            msg = f"{ratio} not in {ratios}."
            raise ValueError(msg)

        # calculations
        self.delta_volume(coral)

        # optimal ratio
        setattr(self, f"{ratio}_optimal", coral)

        # update morphological ratio
        if hasattr(self, f"{ratio}_optimal") and hasattr(coral, ratio):
            return mass_balance(
                getattr(coral, ratio), getattr(self, f"{ratio}_optimal")
            )

    def update(self, coral: Coral):
        """Update morphology.

        :param coral: coral animal
        :type coral: Coral
        """
        # # calculations
        # updated ratios
        ratios = {
            ratio: self.ratio_update(coral, ratio) for ratio in ("rf", "rp", "rs")
        }

        # updated volume
        volume = coral.volume + self.vol_increase

        # update coral morphology
        coral.update_coral_morphology(volume, ratios)
