from src.core.base_model import ExtraModel
from src.core.common.constants import Constants
from src.core.coral.coral_model import Coral
from src.core.common.space_time import CoralOnly


class Recruitment(ExtraModel):
    """Recruitment dynamics."""

    constants: Constants = Constants()

    def update(self, coral: Coral):
        """Update coral cover / volume after spawning event.

        :param coral: coral animal
        :type coral: Coral
        """
        coral.p0[:, 0] += Recruitment.spawning(self, coral, "P")
        coral.update_coral_volume(coral.volume + Recruitment.spawning(self, coral, "V"))

    def spawning(self, coral: Coral, param):
        """Contribution due to mass coral spawning.

        :param coral: coral animal
        :param param: parameter type to which the spawning is added

        :type coral: Coral
        :type param: str
        """
        # # input check
        params = ("P", "V")
        if param not in params:
            msg = f"{param} not in {params}."
            raise ValueError(msg)

        # # calculations
        # potential
        power = 2 if param == "P" else 3
        potential = (
            self.constants.prob_settle
            * self.constants.no_larvae
            * self.constants.d_larvae ** power
        )
        # recruitment
        averaged_healthy_pop = coral.pop_states[:, -1, 0].mean()
        # living cover
        living_cover = coral.RESHAPE.matrix2array(coral.living_cover, "space")

        recruited = CoralOnly().in_space(
            coral=coral,
            function=self.recruited,
            args=(potential, averaged_healthy_pop, living_cover, coral.cover),
        )

        # # output
        return recruited

    @staticmethod
    def recruited(potential, averaged_healthy_pop, cover_real, cover_potential):
        """Determination of recruitment.

        :param potential: recruitment potential
        :param averaged_healthy_pop: model domain averaged healthy population
        :param cover_real: real coral cover
        :param cover_potential: potential coral cover

        :type potential: float
        :type averaged_healthy_pop: float
        :type cover_real: float
        :type cover_potential: float
        """
        return potential * averaged_healthy_pop * (1 - cover_real / cover_potential)
