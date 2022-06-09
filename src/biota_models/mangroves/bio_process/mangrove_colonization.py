from typing import Optional

import numpy as np

from src.biota_models.mangroves.model.mangrove_constants import MangroveConstants
from src.biota_models.mangroves.model.mangrove_model import Mangrove
from src.core.base_model import ExtraModel


class Colonization(ExtraModel):
    """
    Colonization
    1. inundation (max, min water level, flooded only in max waterlevel: intertidal area)
    2. competition
    """
    seed_loc_inun: Optional[np.ndarray]
    seed_loc_shear: Optional[np.ndarray]
    seed_loc: Optional[np.ndarray]

    def update(self, mangrove: Mangrove):
        self.seed_loc = Colonization.colonization_criterion(self, mangrove)

        # cannot establish where mortaility == 1
        loc = np.where(mangrove.mort[self.seed_loc] >= 1)
        np.delete(self.seed_loc, np.where(np.in1d(self.seed_loc, loc) == True))

        # Check if mangroves already present in the cells and if its "full"
        ## TODO
        mangrove.stem_num
        # 3000 individuals per hectare initially!
        ## TODO calculate stem_num depending on grid cell size!
        # determine if already present and full?
        # if no full new can settle but I*C cannot get below 0.5
        # APPEND existing arrays of characteristics!

        ## TODO Columnstack

        mangrove.stem_dia[0, :][self.seed_loc] = mangrove.constants.ini_dia
        mangrove.update_mangrove_characteristics(stem_dia=mangrove.stem_dia)



    def colonization_criterion(self, mangrove: Mangrove):
        """determine areas which are available for colonization

        :param mangrove: mangrove

        :type mangrove: Mangrove
        """
        # inundation
        self.seed_loc_inun = Colonization.colonization_inundation_range(self, mangrove)
        # bed shear stress
        self.seed_loc_shear = Colonization.colonization_shear_stress_criterion(self, mangrove)

        both = np.in1d(self.seed_loc_inun,
                       self.seed_loc_shear)  # array with true in locations where element of a is also in b

        return self.seed_loc_inun[both == True]

    def colonization_inundation_range(self, mangrove: Mangrove):
        """Colonization Inundation range: hydroperiod between 0 and 0.5
        Args:
            mangrove (Mangrove): Mangrove
        """


        return np.where(mangrove.max_wl > 0 & mangrove.inun_rel < 0.5)


    def colonization_shear_stress_criterion(self, mangrove: Mangrove):
        """ Bed shear stress below certain threshold
          Args:
              mangrove (Mangrove): Mangrove
          """

        return np.where(mangrove.max_tau < mangrove.constants.TauThres)


