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
    winter: VegAttribute = False


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
        self.veg_frac = np.zeros(_reshape.space)
        self.veg_frac = self.veg_frac.reshape(len(self.veg_frac), 1)
        self.veg_height = np.zeros(self.veg_frac.shape)
        self.stem_dia = np.zeros(self.veg_frac.shape)
        self.root_len = np.zeros(self.veg_frac.shape)
        self.stem_num = np.zeros(self.veg_frac.shape)
        self.cover = np.zeros(self.veg_frac.shape)
        self.veg_age = np.zeros(self.veg_frac.shape)
        i = self.ls - 1
        self.dt_height = np.zeros((2, 1))

        growth_duration = int((self.constants.get_duration(self.constants.growth_start, self.constants.growth_end)).days)

        if self.ls == 0:
            pass
        elif self.constants.num_ls < self.ls:
            pass
        elif self.ls == 1:
            self.dt_height[0] = (self.constants.maxGrowth_H[i] - self.constants.iniShoot)/growth_duration
            self.dt_height[1] = (self.constants.maxGrowth_H[i] - self.constants.maxH_winter[i]) / growth_duration
            self.dt_stemdia = (self.constants.maxDia[i] - self.constants.iniDia) / (growth_duration * self.constants.maxYears_LS[i])
            self.dt_root = (self.constants.maxRoot[i] - self.constants.iniRoot) / (growth_duration * self.constants.maxYears_LS[i])
        elif self.ls > 1:
            self.dt_height[0] = (self.constants.maxGrowth_H[i] - self.constants.maxH_winter[i]) / growth_duration  # growth per day of growing season
            self.dt_stemdia = (self.constants.maxDia[i] - self.constants.maxDia[i-1]) / (growth_duration * self.constants.maxYears_LS[i])
            self.dt_root = (self.constants.maxRoot[i] - self.constants.maxRoot[i-1]) / (growth_duration * self.constants.maxYears_LS[i])


    def update_growth(self, veg_frac, period, begin_date, end_date):
        """
        update vegetation characteristics based on
        the vegetation age and fraction of veg in each cell (veg_frac_age)
        """
        if self.constants.num_ls < self.ls:
            pass
        else:
            winter_start = pd.to_datetime(self.constants.winter_start).replace(year=begin_date.year)
            start_growth = pd.to_datetime(self.constants.growth_start).replace(year=begin_date.year)
            end_growth = pd.to_datetime(self.constants.growth_end).replace(year=begin_date.year)

            a = start_growth <= pd.to_datetime(period)
            b = pd.to_datetime(period) <= end_growth
            c = np.nonzero((a == True) & (b == True))
            growth_days = len(c[0])


            if begin_date <= winter_start <= end_date:
                self.winter = True
                self.veg_height[veg_frac == 0] = 0 #delete vegetation which died
                self.veg_height[self.constants.maxH_winter[self.ls - 1] < self.veg_height] = self.constants.maxH_winter[self.ls - 1] #change the height for all the vegetation which i biger than max_height_winter to max_height_winter

            else:
                self.winter = False
                self.veg_height[veg_frac > 0] = self.veg_height[veg_frac > 0] + (self.dt_height[0] * growth_days)

            self.stem_dia[veg_frac > 0] = self.stem_dia[veg_frac > 0] + (self.dt_stemdia * growth_days)
            self.root_len[veg_frac > 0] = self.root_len[veg_frac > 0] + (self.dt_root * growth_days)
            self.stem_num[veg_frac > 0] = self.constants.num_stem[self.ls-1]
            self.veg_age[veg_frac > 0] = self.veg_age[veg_frac > 0] + self.constants.ets_duration
            self.cover = veg_frac.sum(axis=1).reshape(-1, 1)



