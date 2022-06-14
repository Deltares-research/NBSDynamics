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

        stem_num_ini = np.zeros(mangrove.ba.shape)  #mangrove.ba surface area [m2]
        stem_num_ini = mangrove.constants.ini_dens * mangrove.ba # initial stem number for each grid cell

        # create new empty arrays for the colonization
        mangrove.stem_num = np.column_stack((np.zeros(mangrove.ba.shape), mangrove.stem_num))
        mangrove.height = np.column_stack((np.zeros(mangrove.ba.shape), mangrove.height))
        mangrove.stem_dia = np.column_stack((np.zeros(mangrove.ba.shape), mangrove.stem_dia))
        mangrove.root_num = np.column_stack((np.zeros(mangrove.ba.shape), mangrove.root_num))

        # determine if already present and full?
        # 1. full --> nothing can settle: delete seed_loc
        full = np.where(np.sum(mangrove.stem_num[self.seed_loc], axis=1) >= stem_num_ini[self.seed_loc])
        np.delete(self.seed_loc, np.where(np.in1d(self.seed_loc, full) == True))

        # 2. empty --> all can settle
        empty = np.where(np.in1d(self.seed_loc, np.where(np.sum(mangrove.stem_num, axis=1) ==0)))
        mangrove.stem_num[:, 0][self.seed_loc[empty]] = stem_num_ini[self.seed_loc[empty]]

        # 3. still space --> some can settle until I * C < 0.5
        space = np.where(np.sum(mangrove.stem_num[self.seed_loc], axis=1) < stem_num_ini[self.seed_loc])
        #solve I*C formular for missing density
        max_C = np.zeros(mangrove.C.shape)
        max_C[np.mean(mangrove.I, axis=1) > 0.5] = 0.51/np.mean(mangrove.I, axis=1)[np.mean(mangrove.I, axis=1) > 0.5]
        tot_possible_B = np.zeros(mangrove.C.shape).reshape(-1,1)
        tot_possible_B[np.mean(mangrove.I, axis=1) > 0.5] = mangrove.B_05.reshape(-1,1)[np.mean(mangrove.I, axis=1) > 0.5]- np.log(1/max_C.reshape(-1,1)[np.mean(mangrove.I, axis=1) > 0.5] - 1)/mangrove.constants.d
        delta_B = tot_possible_B - mangrove.bio_total_cell.reshape(-1, 1)
        W_tot = mangrove.constants.bio_a*mangrove.constants.ini_dia**mangrove.constants.ind_a + mangrove.constants.bio_b*mangrove.constants.ini_dia**mangrove.constants.ind_b
        possible_stem_num = delta_B/W_tot
        mangrove.stem_num[:, 0][space] = possible_stem_num[:, 0][space]


        mangrove.stem_dia[:, 0][mangrove.stem_num[:, 0]>0] = mangrove.constants.ini_dia
        mangrove.update_mangrove_characteristics(col=1, stem_dia=mangrove.stem_dia)



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

        # return np.where(mangrove.max_wl > 0 & mangrove.inun_rel < 0.5)
        a = np.where(0 < mangrove.inun_rel)
        b = np.where(mangrove.inun_rel < 0.5)
        c = np.in1d(a, b)
        return np.delete(a, np.where(c == False))

    def colonization_shear_stress_criterion(self, mangrove: Mangrove):
        """ Bed shear stress below certain threshold
          Args:
              mangrove (Mangrove): Mangrove
          """

        return np.where(mangrove.max_tau < mangrove.constants.TauThres)

