from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from pydantic import validator

import src
from src.core.common.singletons import RESHAPE
from src.core.base_model import ExtraModel
from src.core.common.constants_veg import Constants
from src.core.common.space_time import DataReshape
from src.core.vegetation.veg_only import VegOnly
from src.core.common import fpath_constants_file
from datetime import datetime
from datetime import timedelta
from src.core.vegetation.veg_lifestages import LifeStages

VegAttribute = Union[float, list, tuple, np.ndarray]


class Vegetation(ExtraModel):
    """
    Implements the `VegProtocol`.
    Vegetation object, representing one plant.
    """
    def __init__(self, species):
        super().__init__()
        self.species = species

        self.constants = Constants(species=self.species)

    # other attributes.
        self._cover: Optional[VegAttribute] = list() # sum of fraction of area coverage in each cell (for all ages)
        self.initial = LifeStages(ls=0, constants=self.constants)
        self.juvenile = LifeStages(ls=1, constants=self.constants)
        self.mature = LifeStages(ls=2, constants=self.constants)

    #time related values
    growth_duration: pd.Timedelta = None
    col_duration: pd.Timedelta = None
    winter_duration: pd.Timedelta = None
    # growth_days: VegAttribute = list()
    # growth_Day: VegAttribute = list()
    # col_days: VegAttribute = list()
    # winter_days: VegAttribute = list()

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

# @validator("_cover")
    # @classmethod
    # def validate_vegetation_attribute(
    #    cls, value: Optional[VegAttribute]
    # ) -> Optional[np.ndarray]:
    #    if value is None:
    #        return value
    #    return DataReshape.variable2array(value)

    @property
    def cover(self):  # as input for DFM
        #take cover as sum of all the ages and life stages
        self._cover = self.juvenile.cover + self.mature.cover
        return self.juvenile.cover + self.mature.cover

    @property
    def veg_den(self):  # as input for DFM
        """stem density in number of stems per m2, according to area fraction of veg age"""
        return (self.juvenile.stem_num * self.juvenile.veg_frac).sum(axis=1) + (self.mature.stem_num * self.mature.veg_frac).sum(axis=1)

    @property
    def av_stemdia(self):  # as input for DFM
        """average stem diameter of the different vegetation in one grid cell"""
        return (self.juvenile.stem_dia * self.juvenile.veg_frac).sum(axis=1) / self.juvenile.cover + (self.mature.stem_dia * self.mature.veg_frac).sum(axis=1) / self.mature.cover

    @property
    def av_height(self):  # as input for DFM
        """average shoot height of the different vegetation in one grid cell"""
        return (self.juvenile.veg_height * self.juvenile.veg_frac).sum(axis=1) / self.juvenile.cover + (self.mature.veg_height * self.mature.veg_frac).sum(axis=1) / self.mature.cover

    # def duration_growth(self, constants):
    #     """duration of the growth period from start, end growth from Constants"""
    #     return (constants.get_duration(constants.growth_start, constants.growth_end) / np.timedelta64(1, 'D'))
    #
    # def duration_col(self, constants):
    #     """duration of the colonization period from start, end growth from Constants"""
    #     return (constants.get_duration(constants.ColStart, constants.ColEnd) / np.timedelta64(1, 'D'))
    #
    # def duration_winter(self, constants):
    #     """duration of the colonization period from start, end growth from Constants"""
    #     return (constants.get_duration(constants.winter_start, constants.growth_start) / np.timedelta64(1, 'D'))


    def update_lifestages(self):
        ##TODO CHECK
        #take last colum of previous lifestage and append it in the beginning of new lifestage, delete it from the old lifestage
        if self.initial.veg_frac.all() > 0:
            self.juvenile.veg_frac = np.append(self.initial.veg_frac, self.juvenile.veg_frac, axis=1)
            self.juvenile.veg_height = np.append(self.initial.veg_height, self.juvenile.veg_height, axis=1)
            self.juvenile.stem_dia = np.append(self.initial.stem_dia, self.juvenile.stem_dia, axis=1)
            self.juvenile.root_len = np.append(self.initial.root_len, self.juvenile.root_len, axis=1)
            self.juvenile.stem_num = np.append(self.initial.stem_num, self.juvenile.stem_num, axis=1)
            self.juvenile.veg_age = np.append(self.initial.veg_age, self.juvenile.veg_age, axis=1)
            self.juvenile.cover = self.juvenile.veg_frac.sum(axis=1)
            self.initial.veg_frac = np.zeros(RESHAPE().space)

        if self.juvenile.veg_age > self.constants.maxYears_LS[0]:
            pass

