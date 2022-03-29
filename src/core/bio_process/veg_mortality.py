from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import newton
from src.core import RESHAPE
from src.core.base_model import ExtraModel
from src.core.common.constants_veg import Constants
from src.core.vegetation.veg_model import Vegetation
from random import sample
from src.core.common.space_time import VegOnly, DataReshape
from src.core.vegetation.veg_only import VegOnly
from src.core.hydrodynamics.delft3d import FlowFmModel

class Veg_Mortaility(ExtraModel):
    """Mortality"""

    # drown: Optional[np.array] = None
    # uproot: Optional[np.array] = None
    # burial_scour: Optional[np.array] = None
    # loc_mortality: Optional[np.array] = None
    # constants: Constants = Constants()

    def update(self, veg:Vegetation, constants):
        """Update vegetation characteristics after mortality"""
        veg_loc = np.where(veg.veg_age_frac > 0) #find locations where vegetation is present

        veg.veg_age_frac = veg.veg_age_frac - self.mort_loc(veg, constants) #update fractions due to mortality
        veg.update_vegetation_characteristics(veg.veg_age_frac)

    def mort_loc(self, veg:Vegetation, constants):
        """locations where the vegetation dies"""
        ##TODO see if this makes sense once I know whats in the arrays
        self.loc_mortality = self.mort_criterion(veg, constants)
        return self.loc_mortality

    def mort_criterion(self, veg:Vegetation, constants):
        """criterion for vegetation mortality"""
        # drowning by hydroperiod (flooding thresholds)
        self.drown = np.ones(veg.veg_age_frac.shape)
        self.drown = self.drowing_hydroperiod(veg, constants)
        # uprooting due to high tidal flow velocities (flow velocity thresholds)
        self.uproot = np.ones(veg.veg_age_frac.shape)
        self.uproot = self.uprooting(veg, constants)
        # burial and scour through erosion and sedimentation (dessication thresholds)
        self.burial_scour = np.ones(veg.veg_age_frac.shape)
        self.burial_scour = self.erosion_sedimentation(veg, constants)

        return self.drown + self.uproot + self.burial_scour # something of all three effects combined

    def drowning_hydroperiod(self, veg: Vegetation, constants):

        #find cells that have a hydroperiod/dry period
        #where hydroperiod and dry period is bigger than 0


        return # array with cells where vegetation dies due to HP

    def uprooting(self, veg: Vegetation, constants):
        """
        Mortality through velocity is determined by lin. Function and multiplied with current fraction. For accumulated
        pressure (flooding/drying) the temporal initial fraction at begin of pressure is collected and mortalities are
        linearly reduced from these over several ETS.
        """
        return # array with cells where vegetation dies due to uprooting

    def erosion_sedimentation(self, veg: Vegetation, constants):
        """
        For burial/erosion the length of stem/root is compared with sedimentation/erosion. In case of mortality
        the fraction is set to 0. We assume that the plants adapt to the deposition and erosion rates within one
        ETS by resetting them each ETS.
        """
        shoot_l = veg.veg_height
        root_l = veg.root_len
        return # array with cells where vegetation dies due to burial and scour
