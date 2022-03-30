from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import newton
from src.core import RESHAPE
from src.core.base_model import ExtraModel
from src.core.common.constants_veg import Constants
from src.core.vegetation.veg_model import Vegetation
from src.core.bio_process.veg_hydro_morphodynamics import Hydro_Morphodynamics
from random import sample
from src.core.common.space_time import VegOnly, DataReshape
from src.core.vegetation.veg_only import VegOnly
from src.core.hydrodynamics.delft3d import FlowFmModel

class Veg_Mortality(ExtraModel):
    """Mortality"""

    drown: Optional[np.array] = None
    uproot: Optional[np.array] = None
    burial_scour: Optional[np.array] = None
    loc_mortality: Optional[np.array] = None
    scour: Optional[np.array] = None
    burial: Optional[np.array] = None
    constants: Constants = Constants()
    morph_hydro: Hydro_Morphodynamics = Hydro_Morphodynamics()
    BL_diff: Optional[np.array] = None

    def update(self, veg:Vegetation, constants, ets):
        """Update vegetation characteristics after mortality"""
        veg_loc = np.where(veg.veg_age_frac > 0) #find locations where vegetation is present

        veg.veg_age_frac = veg.veg_age_frac - self.mort_loc(veg, constants, ets) #update fractions due to mortality
        veg.update_vegetation_characteristics(veg.veg_age_frac)

    def mort_loc(self, veg:Vegetation, constants, ets):
        """locations where the vegetation dies"""
        ##TODO see if this makes sense once I know whats in the arrays
        self.loc_mortality = self.mort_criterion(veg, constants, ets)
        return self.loc_mortality

    def mort_criterion(self, veg:Vegetation, constants, ets):
        """criterion for vegetation mortality"""
        # drowning by hydroperiod (flooding thresholds)
        self.drown = np.ones(veg.veg_age_frac.shape)
        self.drown = self.drowing_hydroperiod(veg, constants)
        # uprooting due to high tidal flow velocities (flow velocity thresholds)
        self.uproot = np.ones(veg.veg_age_frac.shape)
        self.uproot = self.uprooting(veg, constants)
        # burial and scour through erosion and sedimentation (dessication thresholds)
        self.burial_scour = np.ones(veg.veg_age_frac.shape)
        self.burial_scour = self.erosion_sedimentation(veg, constants, ets)

        return self.drown + self.uproot + self.burial_scour # something of all three effects combined

    def drowning_hydroperiod(self, veg: Vegetation, constants):

        #find cells that have a hydroperiod/dry period
        #% matrix with dry period during day ?
        #% matrix with hydroperiod

        wet = np.zeros(self.morph_hydro.wl.shape)
        dry = np.zeros(self.morph_hydro.wl.shape)
        wet[self.morph_hydro.wl > 0] = 1
        dry[self.morph_hydro.wl == 0] = 1
        dry = self.morph_hydro.wl_prev*dry #deleting all cells that have fallen wet during this ETS
        wet = self.morph_hydro.wl_prev*wet #deleting all cells that have fallen dry during this ETS
        new_wet = np.where(self.morph_hydro.wl_prev == 0 and self.morph_hydro.wl > 0) #find cells that are newly wet during this ETS
        new_dry = np.where(self.morph_hydro.wl_prev > 0 and self.morph_hydro.wl == 0) #find cells that are newly dry during this ETS
        wet[new_wet] = veg.veg_age_frac[new_wet]  #add initial fractions in cells that are newly wet in matrix
        dry[new_dry] = veg.veg_age_frac[new_dry] #add initial fractions in cells that are newly dry in matrix

        #find cells that are newly flooded during this ETS

        #where hydroperiod and dry period is bigger than 0
        #find slope and threshold for the dying vegetation

        return # array with cells where vegetation dies due to HP

    def flood_dry(self, veg:Vegetation, constants):

        #determine cells that have water depth larger than flooding/drying threshold
        for ls in range(0, constants.num_ls):
            fl = np.zeros(self.morph_hydro.wl.shape)
            fl = np.where(self.morph_hydro.wl > constants.floMort_thres[ls])
            for i in range(0, len(self.morph_hydro.wl[0])):
                if i == 1:
                    flood[fl[:, i]] = 1
                else:
                    temp[fl[:, i]] = 1
                    flood = flood + temp
        #sum up

        constants.desMort_slope
        constants.desMort_thres
        constants.floMort_slope
        constants.floMort_thres

    def uprooting(self, veg: Vegetation, constants):
        """
        Mortality through velocity is determined by lin. Function and multiplied with current fraction. For accumulated
        pressure (flooding/drying) the temporal initial fraction at begin of pressure is collected and mortalities are
        linearly reduced from these over several ETS.
        """
        constants.vel_slope
        constants.vel_thres

        return # array with cells where vegetation dies due to uprooting

    def erosion_sedimentation(self, veg: Vegetation, constants, ets):
        """
        For burial/erosion the length of stem/root is compared with sedimentation/erosion. In case of mortality
        the fraction is set to 0. We assume that the plants adapt to the deposition and erosion rates within one
        ETS by resetting them each ETS.
        """
        self.BedLevel_Dif(veg, ets)
        fract_scour = np.zeros(veg.root_len.shape)
        fract_burial = np.zeros(veg.veg_height)
        for i in range(len(veg.root_len[0])):
            fract_scour[:, i][abs(self.scour) > veg.root_len[:, i]] = 1  #find cells with mortality (scour > rootlength)
            fract_burial[:, i][abs(self.burial) > veg.veg_height[:, i]] = 1

        self.burial_scour = fract_scour + fract_burial  # array with cells where vegetation dies and the fraction of death due to burial and scour

    def BedLevel_Dif(self, veg:Vegetation, ets):
        if ets >= 2:        # from second time step onward in each ets
            if ets == 2:
                self.Bl_diff = veg.bl - veg.bl_prev
            else:
                depth_dts = veg.bl - veg.bl_prev
                self.Bl_diff = depth_dts = self.Bl_diff

            loc_b = np.where(self.Bl_diff < 0)
            self.burial[loc_b] = self.Bl_diff[loc_b]
            loc_s = np.where(self.Bl_diff > 0)
            self.scour[loc_s] = self.Bl_diff[loc_s]
        else:
            self.burial = np.zeros(veg.bl.shape)
            self.scour = np.zeros(veg.bl.shape)



