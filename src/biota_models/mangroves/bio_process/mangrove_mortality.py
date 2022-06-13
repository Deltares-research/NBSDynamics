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


    def update(self, mangrove: Mangrove, ets):
        mort_mark = Mangrove_Mortality.determine_stress(self, mangrove)

        if ets == (self.constants.t_eco_year - 1):
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
                in_s[k] = np.mean(self.I, axis=1) # inundation stress
                in_s_inverse = 1/in_s
                remove = np.zeros(mangrove.stem_num.shape)
                remove = round(mangrove.constants.Mort_plant/(in_s*sum(in_s_inverse)))
                remove[remove > mangrove.stem_num] = mangrove.stem_num
                mangrove.stem_num = mangrove.stem_num - remove
                self.bio_total_cell = Mangrove_Mortality.competition_stress(mangrove) # recalculate total biomass
                mangrove.mort[(self.C*np.mean(self.I, axis=1))>0.5] = 4

                k2 = np.where(np.sum(mort_temp, axis=1) == 1)
                mangrove.stem_num[k2] = mangrove.stem_num[k2] - mangrove.constants.Mort_plant
                Mangrove_Mortality.competition_stress(mangrove)  # recalculate total biomass
                mangrove.mort[(self.C * np.mean(self.I, axis=1)) > 0.5] = 4
                mangrove.C = self.C
                mangrove.I = self.I



    def determine_stress(self, mangrove: Mangrove):

        Mangrove_Mortality.inundation_stress(self, mangrove)
        Mangrove_Mortality.competition_stress(self, mangrove)
        if mangrove.I.ndim > 1:
            Av_I = np.mean(mangrove.I, axis=1)
        else:
            Av_I = mangrove.I

        stress = mangrove.C * Av_I # Ave_I * last_C
        mort_mark = np.zeros(stress.shape)
        mort_mark[stress<=0.5] = 1

        return mort_mark



    def inundation_stress(self, mangrove: Mangrove):
        P = mangrove.inun_rel#relative hydroperiod
        I_current = mangrove.constants.a * P + mangrove.constants.b * P**2 + mangrove.constants.c
        if not mangrove.I:
            mangrove.I = I_current
        else:
            mangrove.I = np.column_stack((mangrove.I, I_current))

    def competition_stress(self, mangrove: Mangrove):

        mangrove.C = 1/(1+ np.exp(mangrove.constants.d*(mangrove.B_05 - mangrove.bio_total_cell)))
