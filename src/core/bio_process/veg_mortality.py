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

    burial_scour: Optional[np.array] = None
    loc_mortality: Optional[np.array] = None
    scour: Optional[np.array] = None
    burial: Optional[np.array] = None
    constants: Constants = Constants()
    morph_hydro: Hydro_Morphodynamics = Hydro_Morphodynamics()
    BL_diff: Optional[np.array] = None
    fraction_dead_flood: Optional[np.array] = None
    fraction_dead_des: Optional[np.array] = None
    fraction_dead_upr: Optional[np.array] = None

    def update(self, veg:Vegetation, constants, ets):
        """Update vegetation characteristics after mortality"""
        self.drowning_hydroperiod(veg, constants)
        self.uprooting(veg, constants)
        self.erosion_sedimentation(veg, constants)

        veg.veg_age_frac = veg.veg_age_frac - self.fraction_dead_flood -self.fraction_dead_des - self.fraction_dead_flood - self.burial_scour #update fractions due to mortality
        veg.veg_age_frac[veg.veg_age_frac < 0] = 0 #replace negative values with 0
        veg.update_vegetation_characteristics(veg.veg_age_frac)

    def drowning_hydroperiod(self, veg: Vegetation, constants):
        flooding_current, drying_current = self.compute_hydroperiod(self.morph_hydro.wl, constants)

        wet = np.zeros(flooding_current.shape)
        dry = np.zeros(drying_current.shape)
        wet[flooding_current > 0] = 1
        dry[drying_current > 0] = 1
        flooding_prev, drying_prev = self.compute_hydroperiod(self.morph_hydro.wl_prev, constants)

        dry = drying_prev*dry #deleting all cells that have fallen wet during this ETS
        wet = flooding_prev*wet #deleting all cells that have fallen dry during this ETS
        new_wet = np.where(flooding_prev == 0 and flooding_current > 0) #find cells that are newly wet during this ETS
        new_dry = np.where(drying_prev == 0 and drying_current > 0) #find cells that are newly dry during this ETS
        for i in range(len(veg.veg_age_frac[0])):
            if i > 0:
                wet = np.column_stack((wet, wet))
                dry = np.column_stack((dry, dry))
            wet[new_wet] = veg.veg_age_frac[:, i][new_wet]  #add initial fractions in cells that are newly wet in matrix
            dry[new_dry] = veg.veg_age_frac[:, i][new_dry] #add initial fractions in cells that are newly dry in matrix

        # determine flooding mortalities based on linear relationship
        mort_flood = np.zeros(veg.veg_age_frac.shape)
        # determine flooding mortalities based on linear relationship
        mort_des =np.zeros(veg.veg_age_frac.shape)

        for ls in range(0, constants.num_ls):
            for d in range(0, constants.maxYears_LS[ls]*sum(veg.growth_days)):
                if d == 0:
                    mort_flood[:, d] = self.mortality_flood_frequency(flooding_current, constants.floMort_thres[ls], constants.floMort_slope[ls])
                    mort_des[:, d] = self.mortality_flood_frequency(drying_current, constants.desMort_thres[ls], constants.desMort_slope[ls])
                else:
                    k = d + sum(constants.maxYears_LS[0:ls]) * sum(veg.growth_days)
                    mort_flood[:, k] = self.mortality_flood_frequency(flooding_current, constants.floMort_thres[ls],
                                                                      constants.floMort_slope[ls])
                    mort_des[:, k] = self.mortality_flood_frequency(drying_current, constants.desMort_thres[ls],
                                                                    constants.desMort_slope[ls])

        self.fraction_dead_flood = mort_flood*wet
        self.fraction_dead_des = mort_des*dry


    @staticmethod
    def compute_hydroperiod(wl_time, constants):
        #determiine cells with water depth > flooding/drying threshold
        fl = np.zeros(wl_time.shape)
        fl = np.where(wl_time > constants.fl_dr)
        flood = np.zeros(wl_time.shape)
        flood[fl] = 1
        flood = flood.sum(axis=1) #sum up for all time steps in the ets

        # compute average flooding and drying period
        flooding_current = np.zeros(flood.shape)
        drying_current = np.zeros(flood.shape)
        flooding_current = flood/constants.ets_duration
        drying_current = (constants.ets_duration-flood)/constants.ets_duration

        return flooding_current, drying_current

    @staticmethod
    def mortality_flood_frequency(fl, th, sl):
        """"
        calculate mortality fraction due to flooding/drying and velocity
        Function: f(x)=sl*x+b for y= 0: b = -sl*th
        th: Threshold
        sl: slope of function
        fl: matrix with flooded days
        """
        b = -th*sl
        dmax = round((1-b)/sl,2) #no. of days when 100% is died off
        fct = (sl*fl+b)  #determines all mortality values over the grid

        out_fl = np.zeros(fl.shape)
        B = np.where(fl>dmax) #cells with 100% mortality
        out_fl[B] = 1
        C = np.where(dmax>fl>th) #cells where fct applies to determine mortality
        out_fl[C] = fct[C]
        return out_fl

    def uprooting(self, veg: Vegetation, constants):
        """
        Mortality through velocity is determined by lin. Function and multiplied with current fraction.
        """
        mort_flow = np.zeros(veg.veg_age_frac.shape)
        for ls in range(0, constants.num_ls):
            for d in range(0, constants.maxYears_LS[ls]*sum(veg.growth_days)):
                if d == 0:
                    mort_flow[:, d] = self.mortality_flood_frequency(veg.max_u, constants.vel_thres[ls], constants.vel_slope[ls])

                else:
                    k = d + sum(constants.maxYears_LS[0:ls]) * sum(veg.growth_days)
                    mort_flow[:, k] = self.mortality_flood_frequency(veg.max_u, constants.vel_thres[ls],
                                                                     constants.vel_slope[ls])

        self.fraction_dead_upr = mort_flow*veg.veg_age_frac


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

    ## TODO make this static method?
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



