from typing import Optional

import numpy as np

from src.biota_models.mangroves.bio_process.mangrove_hydro_morphodynamics import (
    Hydro_Morphodynamics,
)
from src.biota_models.mangroves.model.mangrove_constants import MangroveConstants
from src.biota_models.mangroves.model.mangrove_model import Mangrove
from src.core.base_model import ExtraModel

class Mangrove_Mortality(ExtraModel):
    """Mortality"""

        I: Optional[np.ndarray] = None
        C: Optional[np.ndarray] = None


    def update(self, mangrove: Mangrove):
        # mortality only happens once a year!
        mort_mark = Mangrove_Mortality.determine_stress(mangrove)
        mangrove.mort = mangrove.mort + mort_mark # cells with suppressed growth


        # relative hydroperiod & competition
        # die if I*C <= 0.5
        # average I and C over time

        ## Step 0: Adjust the num to 0 if Inundation <=0.5 & Mort_mark == 5

        while (any(mangrove.mort) == 5):

            mort_temp = np.zeros(mangrove.stem_num.shape)
            mort_temp[mangrove.mort == 5] = 1 #set to one, where mortality status is 5
            mangrove.stem_num[mangrove.mort == 5 and self.I[:, -1] <= 0.5] = 0
            mort_temp[mangrove.stem_num == 0] = 0 #set back to zeros is no vegetation present in cell
            mangrove.mort[mort_temp == 0] = 0

             # Step 2: Kill the mangroves cell by cell

            k = np.where(np.sum(mort_temp, axis=1) > 1)
            in_s = np.zeros(mangrove.stem_num.shape)
            in_s[k] = self.I # inundation stress
            in_s_inverse = 1/in_s
            remove = np.zeros(mangrove.stem_num.shape)
            remove = round(mangrove.constants.Mort_plant/(in_s*sum(in_s_inverse)))
            remove[remove > mangrove.stem_num] = mangrove.stem_num
            mangrove.stem_num = mangrove.stem_num - remove
            self.bio_total_cell = Mangrove_Mortality.competition_stress(mangrove) # recalculate total biomass
            mangrove.mort[(self.C*self.I)>0.5] = 4

            k2 = np.where(np.sum(mort_temp, axis=1) == 1)
            mangrove.stem_num[k2] = mangrove.stem_num[k2] - mangrove.constants.Mort_plant
            self.bio_total_cell = Mangrove_Mortality.competition_stress(mangrove)  # recalculate total biomass
            mangrove.mort[(self.C * self.I) > 0.5] = 4
            mangrove.C = self.C
            mangrove.I = self.I
        # Mangrove.update_mangrove_characteristics(Mangrove_Mortality.I, Mangrove_Mortality.C)


    def determine_stress(self, mangrove: Mangrove):

        Mangrove_Mortality.inundation_stress(mangrove)
        Mangrove_Mortality.competition_stress(mangrove)
        Av_I = np.zeros(self.I.shape)
        for i in range(0, len(self.I)):
            Av_I[i] = np.mean(self.I[i, :])

        stress = self.C * Av_I # Ave_I * last_C
        mort_mark = np.zeros(stress.shape)
        mort_mark[stress<=0.5] = 1

        return mort_mark



    def inundation_stress(self, mangrove:Mangrove):
        P = mangrove.inun_rel#relative hydroperiod
        I_current = mangrove.constants.a * P + mangrove.constants.b * P**2 + mangrove.constants.c
        if not self.I:
            self.I = I_current
        else:
            self.I = np.column_stack((self.I, I_current))

    def competition_stress(self, mangrove:Mangrove):
        W_tree_a = mangrove.constants.bio_a * mangrove.stem_dia**mangrove.constants.ind_a  # aboveground tree weight
        W_tree_b = mangrove.constants.bio_b * mangrove.stem_dia**mangrove.constants.ind_b # belowground tree weight
        B = (W_tree_a + W_tree_b) * mangrove.stem_num

        self.C = 1/(1+ np.exp(mangrove.constants.d*(mangrove.constants.B_05 - B)))

        return B