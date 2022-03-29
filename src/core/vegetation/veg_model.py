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

VegAttribute = Union[float, list, tuple, np.ndarray]


class Vegetation(ExtraModel):
    """
    Implements the `VegProtocol`.
    Vegetation object, representing one plant.
    """

    constants: Constants = Constants()
    veg_height: VegAttribute #vegetation height [m]
    stem_dia: VegAttribute #stem diameter [m]
    veg_den: VegAttribute #vegetation density
    root_len: VegAttribute #root length [m]
    veg_ls: VegAttribute #vegetation life stage (0 or 1 or more), number defined in Constants.num_ls
    veg_age_frac: VegAttribute #vegetation age [yrs]
    stem_num: VegAttribute #number of stems
    ets: VegAttribute #current state of ets, increases over the year (1:constant.ets_per_year)

    # other attributes.
    _cover: Optional[VegAttribute] = None # sum of fraction of area coverage in each cell (for all ages)
    dt_height: VegAttribute = list()
    dt_root: VegAttribute= list()
    dt_stemdia: VegAttribute= list()

    #time related values
    growth_duration: pd.Timedelta
    col_duration: pd.Timedelta
    winter_duration: pd.Timedelta
    growth_days: VegAttribute = list()
    col_days: VegAttribute = list()

    # # hydromorphodynamic environment
    max_tau: Optional[VegAttribute] = None
    max_u: Optional[VegAttribute] = None
    max_wl: Optional[VegAttribute] = None
    min_wl: Optional[VegAttribute] = None
    bl: Optional[VegAttribute] = None


    # @validator("veg_height", "stem_dia", "veg_den", "root_len", "veg_ls", "stem_num")
    # @classmethod
    # def validate_vegetation_attribute(
    #    cls, value: Optional[VegAttribute]
    # ) -> Optional[np.ndarray]:
    #    if value is None:
    #        return value
    #    return DataReshape.variable2array(value)

    def __repr__(self):
        """Development representation."""
        return f"Characteristics({self.veg_height}, {self.stem_dia}, {self.veg_den}, {self.root_len}, {self.veg_ls}, {self.veg_age}, {self.stem_num})"

    def __str__(self):
        """Print representation."""
        return (
            f"Vegetation characteristics with: veg_height = {self.veg_height} m; stem_dia = {self.stem_dia} m;"
            f"veg_den = {self.veg_den} m; veg_root = {self.root_len} m; veg_ls = {self.veg_ls} ;veg_age = {self.veg_age} yrs; stem_num = {self.stem_num} "
        )



    @property
    def veg_den(self):  # as input for DFM
        """stem density in number of stems per m2, according to area fraction of veg age"""
        return (self.stem_num * self.veg_age_frac).sum(axis=1)

    @property
    def av_stemdia(self):  # as input for DFM
        """average stem diameter of the different vegetation in one grid cell"""
        return (self.stem_dia * self.veg_age_frac).sum(axis=1)

    @property
    def av_height(self):  # as input for DFM
        """average shoot height of the different vegetation in one grid cell"""
        return (self.veg_height * self.veg_age_frac).sum(axis=1)

    @property
    def duration_growth(self):
        """duration of the growth period from start, end growth from Constants"""
        return self.get_duration(self.growth_start, self.growth_end)

    @property
    def duration_col(self):
        """duration of the colonization period from start, end growth from Constants"""
        return self.get_duration(self.ColStart, self.ColEnd)

    @property
    def duration_winter(self):
        """duration of the colonization period from start, end growth from Constants"""
        return self.get_duration(self.ColStart, self.ColEnd)


    @property
    def height_matrix(self):
        """self.RESHAPEd vegetation height."""
        return RESHAPE().variable2matrix(self.veg_height, "space")

    @property
    def dia_matrix(self):
        """self.RESHAPEd stem diameter."""
        return RESHAPE().variable2matrix(self.stem_dia, "space")

    @property
    def den_matrix(self):
        """self.RESHAPEd vegetation density."""
        return RESHAPE().variable2matrix(self.veg_den, "space")

    @property
    def root_matrix(self):
        """self.RESHAPEd root length."""
        return RESHAPE().variable2matrix(self.root_len, "space")

    @property
    def ls_matrix(self):
        """self.RESHAPEd vegetation life stage."""
        return RESHAPE().variable2matrix(self.veg_ls, "space")

    @property
    def stemNum_matrix(self):
        """self.RESHAPEd vegetation age."""
        return RESHAPE().variable2matrix(self.stem_num, "space")

    def get_GrowthDays(self):
        """
        find number of growth days in current ets depending on start and end of growth period
        """
        days_ets = 365 / Constants.t_eco_year
        current_date = pd.to_datetime("2022-01-01")
        growth_days = []
        for x in range(0, Constants.t_eco_year):
            growth_day = []
            for y in range(0, round(days_ets)):
                if pd.to_datetime(Constants.growth_start) <= current_date <= pd.to_datetime(Constants.growth_end):
                    growth_day.append(1)
                else:
                    growth_day.append(0)
                current_date = current_date + timedelta(days=1)

            growth_days.append(sum(growth_day))
        return growth_days

    def get_ColWindow(self):
        """
        find ets where colonization happens
        """
        days_ets = 365 / self.constants.t_eco_year
        current_date = pd.to_datetime("2022-01-01")
        col_days = []
        for x in range(0, self.constants.t_eco_year):
            col_day = []
            for y in range(0, round(days_ets)):
                if pd.to_datetime(self.constants.ColStart) <= current_date <= pd.to_datetime(self.constants.ColEnd):
                    col_day.append(1)
                else:
                    col_day.append(0)
                current_date = current_date + timedelta(days=1)

            col_days.append(sum(col_day))
        return col_days


    # @property
    # def cover(self):
    #     """Carrying capacity."""
    #     if self._cover is None:
    #         cover = np.zeros(np.array(self.veg_den).shape)
    #         cover[self.veg_den == 0.0] = 0.0  # 21.09 made 0. instead of just zero
    #         return cover
    #
    #     return self._cover

    def initiate_vegetation_characteristics(self, cover: Optional[np.array]= None):
        _reshape = RESHAPE()
        if cover is not None:
            cover = _reshape.variable2array(cover)
            if not cover.shape[0] == _reshape.space:
                msg = f"Spatial dimension of cover does not match: {cover.shape} =/= {_reshape.space}."
                raise ValueError(msg)
        else:
            cover = np.zeros(_reshape.space)
        self.col_days = self.get_ColWindow()
        self.growth_days = self.get_GrowthDays()
        self.veg_age_frac = np.zeros((len(cover), self.maxAge*sum(self.growth_days)))
        self.veg_age_frac[:, 0] = cover
        self._cover = cover

        # intialization of the vegetation with initial values
        ## TODO change this for other input cases?!
        self.veg_height = np.zeros(self.veg_age_frac.shape)
        self.veg_height[:, 0][cover > 0] = Constants.iniShoot
        self.stem_dia = np.zeros(self.veg_age_frac.shape)
        self.veg_dia[:, 0][cover > 0] = Constants.iniDia
        self.root_len = np.zeros(self.veg_age_frac.shape)
        self.root_len[:, 0][cover > 0] = Constants.iniRoot
        self.veg_ls = np.zeros((len(cover), Constants.num_ls))
        self.veg_ls[:, 0] = cover
        self.stem_num = np.zeros(self.veg_age_frac.shape)
        self.stem_num[:, 0][cover > 0] = Constants.numStem[0]
        self.growth_days = self.get_GrowthDays()


        ## growth slopes of LS
        self.dt_height = np.zeros((Constants.num_ls, Constants.num_ls))
        self.dt_stemdia = np.zeros((Constants.num_ls, Constants.num_ls))
        self.dt_root = np.zeros((Constants.num_ls, Constants.num_ls))
        ## TODO CHECK If this is right!!
        for ls in range(0, Constants.num_ls):
            if ls == 0:
                self.dt_height[0, ls] = (Constants.maxGrowth_H[ls] - Constants.iniShoot)/self.duration_growth(self)
                self.dt_height[1, ls] = (Constants.maxGrowth_H[ls] - Constants.maxH_winter) / self.duration_growth(self)
                self.dt_stemdia[ls] = (Constants.maxDia[ls] - Constants.iniDia) / (self.duration_growth(self) * Constants.maxYears_LS)
                self.dt_root[ls] = (Constants.maxRoot[ls] - Constants.iniRoot) / (self.duration_growth(self) * Constants.maxYears_LS)
            else:
                self.dt_height[0, ls] = (Constants.maxGrowth_H[ls] - Constants.maxH_winter) / self.duration_growth(self)  # growth per day of growing season
                self.dt_height[1, ls] = (Constants.maxGrowth_H[ls] - Constants.maxH_winter) / self.duration_growth(self)  # growth per day of growing season
                self.dt_stemdia[ls] = (Constants.maxDia[ls] - Constants.maxDia[ls-1]) / (self.duration_growth(self) * Constants.maxYears_LS)
                self.dt_root[ls] = (Constants.maxRoot[ls] - Constants.maxRoot[ls-1]) / (self.duration_growth(self) * Constants.maxYears_LS)


    def update_vegetation_characteristics(self, veg_age_frac):
        """
        update vegetation characteristics based on
        the vegetation age and fraction of veg in each cell (veg_frac_age)
        """
        self._cover = veg_age_frac.sum(axis=1) #if this is bigger than one raise an error?!

        for i in range(0, Constants.num_ls): #loop over life stages
            if i == 0:
                self.veg_ls[:, i] = veg_age_frac[:, 0:self.constants.maxYears_LS[i]*sum(self.growth_days)]
            else:
                self.veg_ls[:, i] = veg_age_frac[:, sum(self.constants.maxYears_LS[0:i])*sum(self.growth_days):Constants.maxYears_LS[i]*sum(self.growth_days)].sum(axis=1)

            y = 0
            for j in range(0, Constants.maxYears_LS[i]*sum(self.growth_days)): #loop over all possible days of growth
            # update height, stem diameter, root length based on ets in life stage
                self.stem_num[:, j][self.veg_age_frac[:, j] > 0] = self.constants.numStem[i]  # fraction of ls in cell * stem number of ls

                if j % sum(self.growth_days) == 0: # count years within every life stage
                    y = y+1

                if j == 0 and i == 0: #first column is new vegetation --> ini conditions
                    self.veg_height[:, j][veg_age_frac[:, j] > 0] = self.constants.iniShoot
                    self.stem_dia[:, j][veg_age_frac[:, j] > 0] = self.constants.iniDia
                    self.root_len[:, j][veg_age_frac[:, j] > 0] = self.constants.iniRoot

                elif i == 0 and 0 < j <= sum(self.growth_days): #first year growth starts from ini
                    self.veg_height[:, j][veg_age_frac[:, j] > 0] = self.constants.iniShoot* self.dt_height[0, i]*j
                    self.stem_dia[:, j][veg_age_frac[:, j] > 0] = self.constants.iniDia*self.dt_stemdia[0, i]*j
                    self.root_len[:, j][veg_age_frac[:, j] > 0] = self.constants.iniRoot*self.dt_root[0, i]*j

                elif i == 0 and sum(self.growth_days) < j <= sum(self.growth_days)*self.constants.maxYears_LS[i]:
                    self.veg_height[:, j][veg_age_frac[:, j] > 0] = self.constants.maxH_winter * self.dt_height[1, i]*(j- y*sum(self.growth_days))
                    self.stem_dia[:, j][veg_age_frac[:, j] > 0] = self.constants.iniDia * self.dt_stemdia[0, i]*j
                    self.root_len[:, j][veg_age_frac[:, j] > 0] = self.constants.iniRoot * self.dt_root[0, i]*j

                elif i > 0:
                    k = j + sum(Constants.maxYears_LS[0:i])*sum(self.growth_days)
                    self.veg_height[:, k][veg_age_frac[:, k] > 0] = self.constants.maxH_winter * self.dt_height[0, i]*(j- y*sum(self.growth_days))
                    self.stem_dia[:, k][veg_age_frac[:, k] > 0] = self.constants.iniDia * self.dt_stemdia[0, i]*k
                    self.root_len[:, k][veg_age_frac[:, k] > 0] = self.constants.iniRoot * self.dt_root[0, i] * k
                else:
                    print("Check growth function, there is a further case")

        #update age
        #update fraction
        #update drag coefficient ?

        # USE THIS FUNCTION IN THE MORTALITY, GROWTH & Settlement
        # to determine new vegetation characteristics!
        #     add the new vegetation coming due to colonization
        #     remove vegetation which dies due to mortality


    #def update_lifestage(self):
        #will get values 0 and 1
    #change to 1 in second year


