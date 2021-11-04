from src.core.coral_model import RESHAPE


class Calcification:
    """Calcification rate."""

    def __init__(self, constants):
        """Calcification rate."""
        self.ad = 1
        self.constants = constants

    def calcification_rate(self, coral, omega):
        """Calcification rate.

        :param coral: coral animal
        :param omega: aragonite saturation state

        :type coral: Coral
        :type omega: float, list, tuple, numpy.ndarray
        """

        def aragonite_dependency(calcification_object):
            """Aragonite dependency."""
            calcification_object.ad = (omega - self.constants.omega0) / (
                self.constants.kappaA + omega - self.constants.omega0
            )
            calcification_object.ad = RESHAPE.variable2matrix(
                calcification_object.ad, "time"
            )

        aragonite_dependency(self)
        coral.calc = (
            self.constants.gC
            * coral.Csp
            * coral.pop_states[:, :, 0]
            * self.ad
            * coral.photo_rate
        )
