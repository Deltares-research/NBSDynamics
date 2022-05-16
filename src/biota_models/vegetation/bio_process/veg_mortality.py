from typing import Optional

import numpy as np

from src.biota_models.vegetation.bio_process.veg_hydro_morphodynamics import (
    Hydro_Morphodynamics,
)
from src.biota_models.vegetation.model.veg_constants import VegetationConstants
from src.biota_models.vegetation.model.veg_lifestages import LifeStages
from src.biota_models.vegetation.model.veg_model import Vegetation
from src.core.base_model import ExtraModel


class Veg_Mortality(ExtraModel):
    """Mortality"""

    def __init__(self):
        super().__init__()
        self.burial_scour_j = None
        self.burial_scour_m = None
        self.scour_j = None
        self.burial_j = None
        self.scour_m = None
        self.burial_m = None
        self.Bl_diff = None
        self.fraction_dead_flood_j = None
        self.fraction_dead_des_j = None
        self.fraction_dead_upr_j = None
        self.fraction_dead_flood_m = None
        self.fraction_dead_des_m = None
        self.fraction_dead_upr_m = None

    # burial_scour: Optional[np.array] = None
    # scour: Optional[np.array] = None
    # burial: Optional[np.array] = None
    # BL_diff: Optional[np.array] = None
    # fraction_dead_flood: Optional[np.array] = None
    # fraction_dead_des: Optional[np.array] = None
    # fraction_dead_upr: Optional[np.array] = None

    def update(
        self,
        veg: Vegetation,
        constants: VegetationConstants,
        ets,
        begin_date,
        end_date,
        period,
    ):
        """Update vegetation characteristics after mortality"""
        Veg_Mortality.drowning_hydroperiod(self, veg, constants, ets)
        Veg_Mortality.uprooting(self, veg, constants)
        Veg_Mortality.erosion_sedimentation(self, veg, ets)

        veg.juvenile.veg_frac = (
            veg.juvenile.veg_frac
            - self.fraction_dead_flood_j
            - self.fraction_dead_des_j
            - self.fraction_dead_upr_j
            - self.burial_scour_j
        )  # update fractions due to mortality
        # veg.juvenile.veg_frac[
        #     veg.juvenile.veg_frac < 0
        # ] = 0.0  # replace negative values with 0
        veg.juvenile.veg_frac[
            veg.juvenile.veg_frac < 0.01
        ] = 0.0  # replace negative values and values below 1% coverage with 0
        veg.mature.veg_frac = (
            veg.mature.veg_frac
            - self.fraction_dead_flood_m
            - self.fraction_dead_des_m
            - self.fraction_dead_upr_m
            - self.burial_scour_m
        )  # update fractions due to mortality
        # veg.mature.veg_frac[
        #     veg.mature.veg_frac < 0
        # ] = 0.0  # replace negative values with 0
        veg.mature.veg_frac[
            veg.mature.veg_frac < 0.01
        ] = 0.0  # replace negative values and values below 1% coverage with 0

        veg.juvenile.update_growth(veg.juvenile.veg_frac, period, begin_date, end_date)
        veg.mature.update_growth(veg.mature.veg_frac, period, begin_date, end_date)

        veg.fraction_dead_flood_j = self.fraction_dead_flood_j
        veg.fraction_dead_flood_m = self.fraction_dead_flood_m
        veg.fraction_dead_des_j = self.fraction_dead_des_j
        veg.fraction_dead_des_m = self.fraction_dead_des_m
        veg.fraction_dead_upr_j = self.fraction_dead_upr_j
        veg.fraction_dead_upr_m = self.fraction_dead_upr_m
        veg.burial_scour_j = self.burial_scour_j
        veg.burial_scour_m = self.burial_scour_m

        # veg.total_mort_flood = self.fraction_dead_flood_j + self.fraction_dead_flood_m
        # veg.total_mort_des = self.fraction_dead_des_j + self.fraction_dead_des_m
        # veg.total_mort_upr = self.fraction_dead_upr_j + self.fraction_dead_upr_m
        # veg.total_mort_bursco = self.burial_scour_j + self.burial_scour_m

    def drowning_hydroperiod(
        self, veg: Vegetation, constants: VegetationConstants, ets
    ):
        flooding_current, drying_current = self.compute_hydroperiod(
            veg.wl_ts, constants
        )
        veg.inund = flooding_current

        wet = np.zeros(flooding_current.shape)
        dry = np.zeros(drying_current.shape)
        wet[flooding_current > 0] = 1
        dry[drying_current > 0] = 1
        if ets == 0:
            veg.wl_prev = np.zeros(veg.wl_ts.shape)
        flooding_prev, drying_prev = self.compute_hydroperiod(veg.wl_prev, constants)

        dry = (
            drying_prev * dry
        )  # deleting all cells that have fallen wet during this ETS
        wet = (
            flooding_prev * wet
        )  # deleting all cells that have fallen dry during this ETS

        new_wet = np.where(
            (flooding_prev == 0) & (flooding_current > 0)
        )  # find cells that are newly wet during this ETS
        new_dry = np.where(
            (drying_prev == 0) & (drying_current > 0)
        )  # find cells that are newly dry during this ETS

        wet_j = np.ones(veg.juvenile.veg_frac.shape) * wet.reshape(len(wet), 1)
        # np.repeat(wet.reshape(len(wet), 1), len(veg.juvenile.veg_frac[0]), axis=1)
        dry_j = np.ones(veg.juvenile.veg_frac.shape) * dry.reshape(len(dry), 1)
        wet_j[new_wet] = veg.juvenile.veg_frac[
            new_wet
        ]  # add initial fractions in cells that are newly wet in matrix
        dry_j[new_dry] = veg.juvenile.veg_frac[
            new_dry
        ]  # add initial fractions in cells that are newly dry in matrix
        # determine flooding/drying mortalities based on linear relationship
        mort_flood_j = self.mortality_flood_frequency(
            flooding_current, constants.floMort_thres[0], constants.floMort_slope[0]
        )
        mort_des_j = self.mortality_flood_frequency(
            drying_current, constants.desMort_thres[0], constants.desMort_slope[0]
        )
        self.fraction_dead_flood_j = wet_j * mort_flood_j
        self.fraction_dead_des_j = dry_j * mort_des_j

        wet_m = np.ones(veg.mature.veg_frac.shape) * wet.reshape(len(wet), 1)
        dry_m = np.ones(veg.mature.veg_frac.shape) * dry.reshape(len(dry), 1)
        wet_m[new_wet] = veg.mature.veg_frac[
            new_wet
        ]  # add initial fractions in cells that are newly wet in matrix
        dry_m[new_dry] = veg.mature.veg_frac[
            new_dry
        ]  # add initial fractions in cells that are newly dry in matrix
        # determine flooding/drying mortalities based on linear relationship
        mort_flood_m = self.mortality_flood_frequency(
            flooding_current, constants.floMort_thres[1], constants.floMort_slope[1]
        )
        mort_des_m = self.mortality_flood_frequency(
            drying_current, constants.desMort_thres[1], constants.desMort_slope[1]
        )
        self.fraction_dead_flood_m = wet_m * mort_flood_m
        self.fraction_dead_des_m = dry_m * mort_des_m

    @staticmethod
    def compute_hydroperiod(wl_time, constants: VegetationConstants):
        # determine cells with water depth > flooding/drying threshold
        fl = np.where(wl_time > constants.fl_dr)
        flood = np.zeros(wl_time.shape)
        flood[fl] = 1
        flood = flood.sum(axis=1)  # sum up for all time steps in the ets

        # compute average flooding and drying period
        flooding_current = flood / constants.ets_duration
        drying_current = (constants.ets_duration - flood) / constants.ets_duration

        return flooding_current, drying_current

    @staticmethod
    def mortality_flood_frequency(fl, th, sl):
        """ "
        calculate mortality fraction due to flooding/drying and velocity
        Function: f(x)=sl*x+b for y= 0: b = -sl*th
        th: Threshold
        sl: slope of function
        fl: matrix with flooded days
        """
        b = -th * sl
        dmax = round((1 - b) / sl, 2)  # no. of days when 100% is died off
        fct = sl * fl + b  # determines all mortality values over the grid

        out_fl = np.zeros(fl.shape)
        B = np.where(fl > dmax)  # cells with 100% mortality
        out_fl[B] = 1
        a = fl > th
        b = fl < dmax
        C = np.nonzero(
            (a == True) & (b == True)
        )  # cells where fct applies to determine mortality
        out_fl[C] = fct[C]
        return out_fl.reshape(len(out_fl), 1)

    def uprooting(self, veg: Vegetation, constants: VegetationConstants):
        """
        Mortality through velocity is determined by lin. Function and multiplied with current fraction.
        """
        mort_flow_j = self.mortality_flood_frequency(
            veg.max_u, constants.vel_thres[0], constants.vel_slope[0]
        )
        self.fraction_dead_upr_j = mort_flow_j * veg.juvenile.veg_frac

        mort_flow_m = self.mortality_flood_frequency(
            veg.max_u, constants.vel_thres[1], constants.vel_slope[1]
        )
        self.fraction_dead_upr_m = mort_flow_m * veg.mature.veg_frac

    def erosion_sedimentation(self, veg: Vegetation, ets):
        """
        For burial/erosion the length of stem/root is compared with sedimentation/erosion. In case of mortality
        the fraction is set to 0. We assume that the plants adapt to the deposition and erosion rates within one
        ETS by resetting them each ETS.
        """
        Veg_Mortality.BedLevel_Dif(self, veg, ets)
        fract_scour_j = np.zeros(veg.juvenile.root_len.shape)
        fract_burial_j = np.zeros(veg.juvenile.veg_height.shape)
        fract_scour_j[
            self.scour_j > veg.juvenile.root_len
        ] = 1  # find cells with mortality (scour > rootlength)
        fract_burial_j[self.burial_j > veg.juvenile.veg_height] = 1

        self.burial_scour_j = (
            fract_scour_j + fract_burial_j
        )  # array with cells where vegetation dies and the fraction of death due to burial and scour

        fract_scour_m = np.zeros(veg.mature.root_len.shape)
        fract_burial_m = np.zeros(veg.mature.veg_height.shape)
        fract_scour_m[
            self.scour_m > veg.mature.root_len
        ] = 1  # find cells with mortality (scour > rootlength)
        fract_burial_m[self.burial_m > veg.mature.veg_height] = 1

        self.burial_scour_m = fract_scour_m + fract_burial_m

    ## TODO make this static method?
    def BedLevel_Dif(self, veg: Vegetation, ets):
        if ets >= 1:  # from second time step onward in each ets
            if ets == 1:
                self.Bl_diff = veg.bl - veg.bl_prev
            else:
                depth_dts = veg.bl - veg.bl_prev
                self.Bl_diff = depth_dts + self.Bl_diff

            self.burial_j = np.zeros(veg.juvenile.veg_height.shape)
            self.scour_j = np.zeros(veg.juvenile.root_len.shape)
            self.burial_m = np.zeros(veg.mature.veg_height.shape)
            self.scour_m = np.zeros(veg.mature.root_len.shape)

            loc_b = np.where(self.Bl_diff < 0)
            self.burial_j[loc_b] = (
                np.ones(self.burial_j.shape)
                * self.Bl_diff[0 : len(self.burial_j)].reshape(
                    len(self.Bl_diff[0 : len(self.burial_j)]), 1
                )
            )[loc_b]
            self.burial_m[loc_b] = (
                np.ones(self.burial_m.shape)
                * self.Bl_diff[0 : len(self.burial_m)].reshape(
                    len(self.Bl_diff[0 : len(self.burial_m)]), 1
                )
            )[loc_b]

            loc_s = np.where(self.Bl_diff > 0)
            self.scour_j[loc_s] = (
                np.ones(self.scour_j.shape)
                * self.Bl_diff[0 : len(self.scour_j)].reshape(
                    len(self.Bl_diff[0 : len(self.scour_j)]), 1
                )
            )[loc_s]
            self.scour_m[loc_s] = (
                np.ones(self.scour_m.shape)
                * self.Bl_diff[0 : len(self.scour_m)].reshape(
                    len(self.Bl_diff[0 : len(self.scour_m)]), 1
                )
            )[loc_s]

        else:
            self.burial_j = np.zeros(veg.juvenile.veg_height.shape)
            self.scour_j = np.zeros(veg.juvenile.root_len.shape)
            self.burial_m = np.zeros(veg.mature.veg_height.shape)
            self.scour_m = np.zeros(veg.mature.root_len.shape)
