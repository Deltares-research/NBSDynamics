from typing import Optional
import numpy as np
from src.core.base_model import ExtraModel
from src.core.vegetation.veg_model import Vegetation
from src.core.common.constants_veg import Constants


class Veg_Growth(ExtraModel):
    """ Vegetation growth by changing the veg_frac_age matrix"""

    growth_days_ets: Optional[float] = None

    def update(self, veg: Vegetation, ets):
        """update vegetation depending on growth days in certain ets"""
        self.growth_days_ets = veg.growth_days[ets]
        n, m = veg.veg_age_frac.shape
        new = np.zeros((n, self.growth_days_ets))
        veg.veg_age_frac = np.hstack((new, veg.veg_age_frac))
        # sum up all vegetation that is older than maxAge and the left over columns
        veg.veg_age_frac[:, Constants.maxAge*sum(veg.growth_days)] = veg.veg_age_frac[:,Constants.maxAge*sum(veg.growth_days):-1].sum(axis=1)
        veg.veg_age_frac = np.delete(veg.veg_age_frac, np.s_[Constants.maxAge*sum(veg.growth_days):-1], axis=1)
        veg.veg_age_frac = np.delete(veg.veg_age_frac, -1, axis=1)
        #veg.veg_age_frac[veg.veg_age_frac[:, Constants.maxAge*sum(veg.growth_days)+1:-1]>0] = 0

        veg.update_vegetation_characteristics(veg.veg_age_frac)
