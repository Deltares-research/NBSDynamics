from src.core import RESHAPE
from src.core.common.constants import Constants
from src.core.biota.coral.coral_model import Coral


class Temperature:
    def __init__(self, temperature, constants: Constants = Constants()):
        """
        Thermal micro-environment.

        Parameters
        ----------
        temperature : numeric
            Temperature of water [K].
        """
        self.T = RESHAPE().variable2matrix(temperature, "time")
        self.constants = constants

    def coral_temperature(self, coral: Coral):
        """Coral temperature.

        :param coral: coral animal
        :type coral: Coral
        """
        if self.constants.tme:
            delta_t = RESHAPE().variable2matrix(coral.delta_t, "space")
            coral.dTc = (
                (delta_t * self.constants.ap)
                / (self.constants.k * self.constants.K0)
                * coral.light
            )
            coral.temp = self.T + coral.dTc
        else:
            coral.temp = self.T
