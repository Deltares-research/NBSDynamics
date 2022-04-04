from typing import Dict, Optional, Union

import numpy as np
from src.core.common.singletons import RESHAPE
from src.core.base_model import ExtraModel
import datetime
import pandas as pd

VegAttribute = Union[float, list, tuple, np.ndarray]

class LifeStages(ExtraModel):

    def __init__(self, ls, constants):
        super().__init__()
        self.ls = ls
        self.constants = constants
    veg_height: VegAttribute = None # vegetation height [m]
    stem_dia: VegAttribute = None # stem diameter [m]
    root_len: VegAttribute = None # root length [m]
    veg_age: VegAttribute = None  # vegetation life stage (0 or 1 or more), number defined in Constants.num_ls
    veg_frac: VegAttribute = None  # vegetation age [yrs]
    stem_num: VegAttribute = None  # number of stems
    cover: VegAttribute = None #vegetation fraction of all ages

    dt_height: VegAttribute = list()
    dt_root: VegAttribute = list()
    dt_stemdia: VegAttribute = list()


    def __repr__(self):
        """Development representation."""
        return f"Characteristics({self.veg_height}, {self.stem_dia}, {self.root_len}, {self.veg_age}, {self.stem_num}, {self.cover})"

    def __str__(self):
        """Print representation."""
        return (
            f"Vegetation characteristics with: veg_height = {self.veg_height} m; stem_dia = {self.stem_dia} m; veg_root = {self.root_len} m; veg_age = {self.veg_age} days; stem_num = {self.stem_num}; "
            f"cover = {self.cover}"
        )

    #
    # @validator("veg_height", "stem_dia",  "root_len", "stem_num", "cover")
    # @classmethod
    # def validate_vegetation_attribute(
    #    cls, value: Optional[VegAttribute]
    # ) -> Optional[np.ndarray]:
    #    if value is None:
    #        return value
    #    return DataReshape.variable2array(value)


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
    # def root_matrix(self):
    #     """self.RESHAPEd root length."""
    #     return RESHAPE().variable2matrix(self.root_len, "space")
    #
    # @property
    # def stemNum_matrix(self):
    #     """self.RESHAPEd vegetation age."""
    #     return RESHAPE().variable2matrix(self.stem_num, "space")



    def initiate_vegetation_characteristics(self):
        _reshape = RESHAPE()

        # intialization of the vegetation with initial values
        ## TODO change this for other input cases with initial cover!
        self.veg_height = np.zeros(_reshape.space)
        self.stem_dia = np.zeros(_reshape.space)
        self.root_len = np.zeros(_reshape.space)
        self.stem_num = np.zeros(_reshape.space)
        self.cover = np.zeros(_reshape.space)
        self.veg_frac = np.zeros(_reshape.space)
        self.veg_age = np.zeros(_reshape.space)
        i = self.ls - 1
        self.dt_height = np.zeros((2, 1))

        if self.ls == 0:
            pass
        elif self.ls == 1:
            self.dt_height[i] = (self.constants.maxGrowth_H[i] - self.constants.iniShoot)/(sum(self.constants.growth_days))
            self.dt_height[i] = (self.constants.maxGrowth_H[i] - self.constants.maxH_winter[i]) / sum(self.constants.growth_days)
            self.dt_stemdia = (self.constants.maxDia[i] - self.constants.iniDia) / sum(self.constants.growth_days) * self.constants.maxYears_LS[i]
            self.dt_root = (self.constants.maxRoot[i] - self.constants.iniRoot) / sum(self.constants.growth_days) * self.constants.maxYears_LS[i]
        elif self.ls > 1:
            self.dt_height[i] = (self.constants.maxGrowth_H[i] - self.constants.maxH_winter[i]) / (sum(self.constants.growth_days))  # growth per day of growing season
            self.dt_stemdia = (self.constants.maxDia[i] - self.constants.maxDia[i-1]) / (sum(self.constants.growth_days) * self.constants.maxYears_LS[i])
            self.dt_root = (self.constants.maxRoot[i] - self.constants.maxRoot[i-1]) / (sum(self.constants.growth_days) * self.constants.maxYears_LS[i])





    # def update_winter(self, veg_frac):
    #     self.veg_frac = veg_frac
    #     self.veg_height[veg_frac > 0] = self.constants.maxH_winter[self.ls]
    #     self.stem_dia[veg_frac > 0] = self.stem_dia
    #     self.root_len[veg_frac > 0] = self.root_len
    #     self.stem_num[veg_frac > 0] = self.stem_num
    #     self.veg_age[veg_frac > 0] = self.veg_age + self.constants.ets_duration()
    #     self.cover = veg_frac.sum(axis=1)

    def update_growth(self, veg_frac, ets, begin_date, end_date):
        """
        update vegetation characteristics based on
        the vegetation age and fraction of veg in each cell (veg_frac_age)
        """
        self.constants.winter_start = pd.to_datetime(self.constants.winter_start)
        self.constants.winter_start = self.constants.winter_start.replace(year=begin_date.year)

        if begin_date <= self.constants.winter_start <= end_date:
            self.veg_height[veg_frac > 0] = self.constants.maxH_winter[self.ls]
        else:
            self.veg_height[veg_frac > 0] = self.veg_height[veg_frac > 0] + self.dt_height[0] * self.constants.growth_days[ets]
        self.stem_dia[veg_frac > 0] = self.stem_dia[veg_frac > 0] + self.dt_stemdia * self.constants.growth_days[ets]
        self.root_len[veg_frac > 0] = self.root_len[veg_frac > 0] + self.dt_root * self.constants.growth_days[ets]
        self.stem_num[veg_frac > 0] = self.constants.num_stem[self.ls-1]
        self.veg_age[veg_frac > 0] = self.veg_age[veg_frac > 0] + self.constants.ets_duration
        m = veg_frac.shape
        if len(m) > 1:
            self.cover = veg_frac.sum(axis=1)



