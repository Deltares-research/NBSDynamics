import json
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from pydantic import validator

from src.core.common.singletons import RESHAPE
from src.core.base_model import ExtraModel
from src.core.common.constants_veg import Constants
from src.core.common.space_time import DataReshape
from src.core.vegetation.veg_only import VegOnly
from src.core.common import fpath_constants_file
from datetime import datetime
from datetime import timedelta
from src.core.vegetation.veg_model import LifeStages

VegAttribute = Union[float, list, tuple, np.ndarray]


class Vegetation(ExtraModel):
    """
    Implements the `VegProtocol`.
    Vegetation object, representing one plant.
    """

    constants: Constants = Constants()

    # other attributes.
    _cover: Optional[VegAttribute] = list() # sum of fraction of area coverage in each cell (for all ages)
    initial = LifeStages(ls = 0)
    juvenile = LifeStages(ls = 1)
    mature = LifeStages(ls =2)

    #time related values
    growth_duration: pd.Timedelta
    col_duration: pd.Timedelta
    winter_duration: pd.Timedelta
    growth_days: VegAttribute = list()
    growth_Day: VegAttribute = list()
    col_days: VegAttribute = list()

    # # hydromorphodynamic environment
    max_tau: Optional[VegAttribute] = None
    max_u: Optional[VegAttribute] = None
    max_wl: Optional[VegAttribute] = None
    min_wl: Optional[VegAttribute] = None
    bl: Optional[VegAttribute] = None
    max_tau_prev: Optional[VegAttribute] = None
    max_u_prev: Optional[VegAttribute] = None
    max_wl_prev: Optional[VegAttribute] = None
    min_wl_prev: Optional[VegAttribute] = None
    bl_prev: Optional[VegAttribute] = None
    wl_prev: Optional[VegAttribute] = None
    tau_ts: Optional[VegAttribute] = None
    u_ts: Optional[VegAttribute] = None
    wl_ts: Optional[VegAttribute] = None
    bl_ts: Optional[VegAttribute] = None

    # @validator("veg_height", "stem_dia", "veg_den", "root_len", "veg_ls", "stem_num")
    # @classmethod
    # def validate_vegetation_attribute(
    #    cls, value: Optional[VegAttribute]
    # ) -> Optional[np.ndarray]:
    #    if value is None:
    #        return value
    #    return DataReshape.variable2array(value)
    @property
    def cover(self):  # as input for DFM
        """average shoot height of the different vegetation in one grid cell"""

        #take cover as sum of all the ages and life stages

        return self._cover

    @property
    def veg_den(self):  # as input for DFM
        """stem density in number of stems per m2, according to area fraction of veg age"""
        return (self.stem_num * self.veg_age).sum(axis=1)

    @property
    def av_stemdia(self):  # as input for DFM
        """average stem diameter of the different vegetation in one grid cell"""
        return (self.stem_dia * self.veg_age_frac).sum(axis=1) / self._cover

    @property
    def av_height(self):  # as input for DFM
        """average shoot height of the different vegetation in one grid cell"""
        return (self.veg_height * self.veg_age_frac).sum(axis=1) / self._cover

    def duration_growth(self, constants):
        """duration of the growth period from start, end growth from Constants"""
        return (constants.get_duration(constants.growth_start, constants.growth_end) / np.timedelta64(1, 'D'))

    def duration_col(self, constants):
        """duration of the colonization period from start, end growth from Constants"""
        return (constants.get_duration(constants.ColStart, constants.ColEnd) / np.timedelta64(1, 'D'))

    def duration_winter(self, constants):
        """duration of the colonization period from start, end growth from Constants"""
        return (constants.get_duration(constants.winter_start, constants.growth_start) / np.timedelta64(1, 'D'))

    def get_GrowthDays(self, constants):
        """
        find number of growth days in current ets depending on start and end of growth period
        """
        current_date = pd.to_datetime(constants.start_date)
        growth_days = []
        for x in range(0, constants.t_eco_year):
            growth_Day = []
            for y in range(0, round(constants.ets_duration)):
                if pd.to_datetime(constants.growth_start).month <= current_date.month <= pd.to_datetime(
                        constants.growth_end).month:
                    if pd.to_datetime(constants.growth_start).month == current_date.month:
                        if pd.to_datetime(constants.growth_start).day <= current_date.day:
                            growth_Day.append(1)
                        else:
                            growth_Day.append(0)
                    elif pd.to_datetime(constants.growth_end).month == current_date.month:
                        if current_date.day <= pd.to_datetime(constants.growth_end).day:
                            growth_Day.append(1)
                        else:
                            growth_Day.append(0)
                    else:
                        growth_Day.append(1)
                else:
                    growth_Day.append(0)
                current_date = current_date + timedelta(days=1)

            growth_days.append(sum(growth_Day))
        return np.array(growth_days)

    def get_ColWindow(self, constants):
        """
        find ets where colonization happens
        """
        days_ets = 365 / constants.t_eco_year
        current_date = pd.to_datetime(constants.start_date)
        col_days = []
        for x in range(0, constants.t_eco_year):
            col_Day = []
            for y in range(0, round(days_ets)):
                if pd.to_datetime(constants.ColStart).month <= current_date.month <= pd.to_datetime(
                        constants.ColEnd).month:
                    if pd.to_datetime(constants.ColStart).month == current_date.month:
                        if pd.to_datetime(constants.ColStart).day <= current_date.day:
                            col_Day.append(1)
                        else:
                            col_Day.append(0)
                    elif pd.to_datetime(constants.ColEnd).month == current_date.month:
                        if current_date.day <= pd.to_datetime(constants.ColEnd).day:
                            col_Day.append(1)
                        else:
                            col_Day.append(0)
                    else:
                        col_Day.append(1)
                else:
                    col_Day.append(0)
                current_date = current_date + timedelta(days=1)

            col_days.append(sum(col_Day))
        return np.array(col_days)

    @staticmethod
    def update_lifestages(ls0, ls1, ls2=None):
        #take last colum of previous lifestage and append it in the beginning of new lifestage, delete it from the old lifestage


class LifeStages(Vegetation):

    def __init__(self, ls):

    veg_height: VegAttribute  # vegetation height [m]
    stem_dia: VegAttribute  # stem diameter [m]
    root_len: VegAttribute  # root length [m]
    veg_age: VegAttribute  # vegetation life stage (0 or 1 or more), number defined in Constants.num_ls
    veg_frac: VegAttribute  # vegetation age [yrs]
    stem_num: VegAttribute  # number of stems
    constants: Constants = Constants()

    dt_height: VegAttribute = list()
    dt_root: VegAttribute = list()
    dt_stemdia: VegAttribute = list()

    #
    # def __repr__(self):
    #     """Development representation."""
    #     return f"Characteristics({self.veg_height}, {self.stem_dia}, {self.veg_den}, {self.root_len}, {self.veg_ls}, {self.veg_age}, {self.stem_num})"
    #
    # def __str__(self):
    #     """Print representation."""
    #     return (
    #         f"Vegetation characteristics with: veg_height = {self.veg_height} m; stem_dia = {self.stem_dia} m;"
    #         f"veg_den = {self.veg_den} m; veg_root = {self.root_len} m; veg_ls = {self.veg_ls} ;veg_age = {self.veg_age} yrs; stem_num = {self.stem_num} "
    #     )
    #
    #

    # @property
    # def height_matrix(self):
    #     """self.RESHAPEd vegetation height."""
    #     return RESHAPE().variable2matrix(self.veg_height, "space")
    #
    # @property
    # def dia_matrix(self):
    #     """self.RESHAPEd stem diameter."""
    #     return RESHAPE().variable2matrix(self.stem_dia, "space")
    #
    # @property
    # def den_matrix(self):
    #     """self.RESHAPEd vegetation density."""
    #     return RESHAPE().variable2matrix(self.veg_den, "space")
    #
    # @property
    # def root_matrix(self):
    #     """self.RESHAPEd root length."""
    #     return RESHAPE().variable2matrix(self.root_len, "space")
    #
    # @property
    # def ls_matrix(self):
    #     """self.RESHAPEd vegetation life stage."""
    #     return RESHAPE().variable2matrix(self.veg_ls, "space")
    #
    # @property
    # def stemNum_matrix(self):
    #     """self.RESHAPEd vegetation age."""
    #     return RESHAPE().variable2matrix(self.stem_num, "space")


    # @property
    # def cover(self):
    #     """Carrying capacity."""
    #     if self._cover is None:
    #         cover = np.zeros(np.array(self.veg_den).shape)
    #         cover[self.veg_den == 0.0] = 0.0  # 21.09 made 0. instead of just zero
    #         return cover
    #
    #     return self._cover

    def initiate_vegetation_characteristics(self, ls):
        _reshape = RESHAPE()


        # intialization of the vegetation with initial values
        ## TODO change this for other input cases?!
        self.veg_height = np.zeros(_reshape.space)
        self.stem_dia = np.zeros(_reshape.space)
        self.root_len = np.zeros(_reshape.space)
        self.stem_num = np.zeros(_reshape.space)

        if ls == 0:
            self.dt_height[ls] = (LifeStages.constants.maxGrowth_H[ls] - constants.iniShoot)/(self.duration_growth(self, constants))
            self.dt_height[ls] = (constants.maxGrowth_H[ls] - constants.maxH_winter[ls]) / (self.duration_growth(self, constants))
            self.dt_stemdia = (constants.maxDia[ls] - constants.iniDia) / ((self.duration_growth(self, constants)) * constants.maxYears_LS[ls])
            self.dt_root = (constants.maxRoot[ls] - constants.iniRoot) / ((self.duration_growth(self, constants)) * constants.maxYears_LS[ls])
        else:
            self.dt_height[ls] = (constants.maxGrowth_H[ls] - constants.maxH_winter[ls]) / (self.duration_growth(self, constants))  # growth per day of growing season
            self.dt_height[ls] = (constants.maxGrowth_H[ls] - constants.maxH_winter[ls]) / self.duration_growth(self, constants)  # growth per day of growing season
            self.dt_stemdia = (constants.maxDia[ls] - constants.maxDia[ls-1]) / (self.duration_growth(self, constants) * constants.maxYears_LS[ls])
            self.dt_root = (constants.maxRoot[ls] - constants.maxRoot[ls-1]) / (self.duration_growth(self, constants) * constants.maxYears_LS[ls])

    def update_growth(self, veg_frac):

    def update_nogrowth(self, veg_frac):



    # def update_vegetation_characteristics(self, veg_age_frac, constants):
    #     """
    #     update vegetation characteristics based on
    #     the vegetation age and fraction of veg in each cell (veg_frac_age)
    #     """
    #     self._cover = veg_age_frac.sum(axis=1) #if this is bigger than one raise an error?!
    #
    #     for i in range(0, constants.num_ls): #loop over life stages
    #         if i == 0: ## TODO how to get sum from several columns
    #             self.veg_ls[:, i] = veg_age_frac[:, 0:(constants.maxYears_LS[i]*sum(self.growth_days)+1)].sum(axis=1)
    #         else:
    #             self.veg_ls[:, i] = veg_age_frac[:, sum(constants.maxYears_LS[0:i])*sum(self.growth_days):constants.maxYears_LS[i]*sum(self.growth_days)].sum(axis=1)
    #
    #         y = 0
    #         for j in range(0, constants.maxYears_LS[i]*sum(self.growth_days)): #loop over all possible days of growth
    #         # update height, stem diameter, root length based on ets in life stage
    #             if j % sum(self.growth_days) == 0:
    #                 y = y + 1 # count years within every life stage
    #
    #             if j == 0 and i == 0: #first column is new vegetation --> ini conditions
    #                 self.stem_num[:, j] = constants.numStem[i] * self.veg_age_frac[:, j]  # fraction of ls in cell * stem number of l
    #                 self.veg_height[:, j][veg_age_frac[:, j] > 0] = constants.iniShoot
    #                 self.stem_dia[:, j][veg_age_frac[:, j] > 0] = constants.iniDia
    #                 self.root_len[:, j][veg_age_frac[:, j] > 0] = constants.iniRoot
    #
    #             elif i == 0 and 0 < j <= sum(self.growth_days): #first year growth starts from ini (seedling)
    #                 self.stem_num[:, j] = constants.numStem[i] * self.veg_age_frac[:, j]  # fraction of ls in cell * stem number of ls
    #                 self.veg_height[:, j][veg_age_frac[:, j] > 0] = constants.iniShoot + self.dt_height[0, i] * j
    #                 self.stem_dia[:, j][veg_age_frac[:, j] > 0] = constants.iniDia + self.dt_stemdia[0, i] * j
    #                 self.root_len[:, j][veg_age_frac[:, j] > 0] = constants.iniRoot + self.dt_root[0, i] * j
    #
    #             elif i == 0 and sum(self.growth_days) < j <= sum(self.growth_days)*constants.maxYears_LS[i]: #if first life stage (seedling) is longer than 1 year
    #                 self.stem_num[:, j] = constants.numStem[i] * self.veg_age_frac[:, j]  # fraction of ls in cell * stem number of ls
    #                 self.veg_height[:, j][veg_age_frac[:, j] > 0] = constants.maxH_winter[i] + self.dt_height[1, i]*(j - y*sum(self.growth_days))
    #                 self.stem_dia[:, j][veg_age_frac[:, j] > 0] = constants.iniDia + self.dt_stemdia[0, i]*j
    #                 self.root_len[:, j][veg_age_frac[:, j] > 0] = constants.iniRoot + self.dt_root[0, i]*j
    #
    #             elif i > 0:
    #                 k = j + sum(constants.maxYears_LS[0:i])*sum(self.growth_days)
    #                 self.stem_num[:, k] = constants.numStem[i] * self.veg_age_frac[:, k]  # fraction of ls in cell * stem number of ls
    #                 self.veg_height[:, k][veg_age_frac[:, k] > 0] = constants.maxH_winter[i] + self.dt_height[0, i] * (j - y * sum(self.growth_days))
    #                 self.stem_dia[:, k][veg_age_frac[:, k] > 0] = constants.maxDia[i-1] + self.dt_stemdia[0, i] * j
    #                 self.root_len[:, k][veg_age_frac[:, k] > 0] = constants.maxRoot[i-1] + self.dt_root[0, i] * j
    #
    #             else:
    #                 print("NO, something went wrong! Check growth function, there is a further case")
    #

