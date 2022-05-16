from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import root_validator

from src.biota_models.vegetation.model.veg_constants import VegetationConstants
from src.biota_models.vegetation.model.veg_lifestages import LifeStages
from src.core.biota.biota_model import Biota
from src.core.common.singletons import RESHAPE
from src.biota_models.vegetation.bio_process.veg_mortality import Veg_Mortality

VegAttribute = Union[float, list, tuple, np.ndarray]


class Vegetation(Biota):
    """
    Implements the `VegProtocol`.
    Vegetation object, representing one plant.
    """

    species: str
    constants: Optional[VegetationConstants]
    total_cover: Optional[
        VegAttribute
    ]  # sum of fraction of area coverage in each cell (for all ages)
    initial: Optional[LifeStages]
    juvenile: Optional[LifeStages]
    mature: Optional[LifeStages]

    @root_validator
    def configure_veg_fields(cls, values: dict) -> dict:
        veg_constants = values.get("constants", None)
        if not veg_constants:
            veg_constants = VegetationConstants(species=values["species"])
            values["constants"] = veg_constants

        # other attributes.
        values["initial"] = LifeStages(ls=0, constants=veg_constants)
        values["juvenile"] = LifeStages(ls=1, constants=veg_constants)
        values["mature"] = LifeStages(ls=2, constants=veg_constants)
        if not values.get("total_cover", None):
            # Initialize as an empty list if it  has not been done earlier.
            values["total_cover"] = []
        return values

    # time related values
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
    inund: Optional[VegAttribute] = None

    # @validator("_cover")
    # @classmethod
    # def validate_vegetation_attribute(
    #    cls, value: Optional[VegAttribute]
    # ) -> Optional[np.ndarray]:
    #    if value is None:
    #        return value
    #    return DataReshape.variable2array(value)

    @property
    def total_cover(self):  # as input for DFM
        # take cover as sum of all the ages and life stages
        # self.total_cover = self.juvenile.cover + self.mature.cover
        return self.juvenile.cover + self.mature.cover

    @property
    def veg_den(self):  # as input for DFM
        """stem density in number of stems per m2, according to area fraction of veg age"""
        return (self.juvenile.stem_num * self.juvenile.veg_frac).sum(axis=1) + (
            self.mature.stem_num * self.mature.veg_frac
        ).sum(axis=1)

    @property
    def av_stemdia(self):  # as input for DFM
        """average stem diameter of the different vegetation in one grid cell"""
        cover_j = self.juvenile.cover.copy()
        cover_m = self.mature.cover.copy()
        cover_j[cover_j == 0] = 1
        cover_m[cover_m == 0] = 1
        # if self.juvenile.cover.all() == 0 and self.mature.cover.all() == 0:
        #     return np.zeros(self.cover.shape)
        # elif self.mature.cover.all() == 0:
        #     return (self.juvenile.stem_dia * self.juvenile.veg_frac).sum(axis=1) / self.juvenile.cover
        # elif self.juvenile.cover.all() == 0:
        #     return (self.mature.stem_dia * self.mature.veg_frac).sum(axis=1) / self.mature.cover
        # else:
        return (self.juvenile.stem_dia * self.juvenile.veg_frac).sum(axis=1).reshape(
            -1, 1
        ) / cover_j + (self.mature.stem_dia * self.mature.veg_frac).sum(axis=1).reshape(
            -1, 1
        ) / cover_m

    @property
    def av_height(self):  # as input for DFM
        """average shoot height of the different vegetation in one grid cell"""
        cover_j = self.juvenile.cover.copy()
        cover_m = self.mature.cover.copy()
        cover_j[cover_j == 0] = 1
        cover_m[cover_m == 0] = 1
        # if np.all(self.juvenile.cover == 0) and np.all(self.mature.cover == 0):
        #     return np.zeros(self.cover.shape)
        # elif np.all(self.mature.cover == 0):
        #     return (self.juvenile.veg_height * self.juvenile.veg_frac).sum(axis=1) / (self.juvenile.cover[self.juvenile.cover == 0]=1)
        # elif self.juvenile.cover.all() == 0:
        #     return (self.mature.veg_height * self.mature.veg_frac).sum(axis=1) / self.mature.cover
        # else:
        return (self.juvenile.veg_height * self.juvenile.veg_frac).sum(axis=1).reshape(
            -1, 1
        ) / cover_j + (self.mature.veg_height * self.mature.veg_frac).sum(
            axis=1
        ).reshape(
            -1, 1
        ) / cover_m

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
        _reshape = RESHAPE()
        # take last colum of previous lifestage and append it in the beginning of new lifestage, delete it from the old lifestage
        if np.any(self.initial.veg_frac > 0):
            self.juvenile.veg_frac = np.column_stack(
                (self.initial.veg_frac, self.juvenile.veg_frac)
            )
            self.juvenile.veg_height = np.column_stack(
                (self.initial.veg_height, self.juvenile.veg_height)
            )
            self.juvenile.stem_dia = np.column_stack(
                (self.initial.stem_dia, self.juvenile.stem_dia)
            )
            self.juvenile.root_len = np.column_stack(
                (self.initial.root_len, self.juvenile.root_len)
            )
            self.juvenile.stem_num = np.column_stack(
                (self.initial.stem_num, self.juvenile.stem_num)
            )
            self.juvenile.veg_age = np.column_stack(
                (self.initial.veg_age, self.juvenile.veg_age)
            )

            # empty initial arrays
            self.initial.veg_frac = np.zeros(self.initial.veg_height.shape)
            self.initial.veg_height = np.zeros(self.initial.veg_frac.shape)
            self.initial.stem_dia = np.zeros(self.initial.veg_height.shape)
            self.initial.root_len = np.zeros(self.initial.veg_height.shape)
            self.initial.stem_num = np.zeros(self.initial.veg_height.shape)
            self.juvenile.veg_frac = np.delete(
                self.juvenile.veg_frac,
                np.where(np.all(self.juvenile.veg_frac == 0, axis=0) == True),
                1,
            )
            self.juvenile.veg_height = np.delete(
                self.juvenile.veg_height,
                np.where(np.all(self.juvenile.veg_height == 0, axis=0) == True),
                1,
            )
            self.juvenile.veg_age = np.delete(
                self.juvenile.veg_age,
                np.where(np.all(self.juvenile.stem_dia == 0, axis=0) == True),
                1,
            )
            self.juvenile.stem_dia = np.delete(
                self.juvenile.stem_dia,
                np.where(np.all(self.juvenile.stem_dia == 0, axis=0) == True),
                1,
            )
            self.juvenile.root_len = np.delete(
                self.juvenile.root_len,
                np.where(np.all(self.juvenile.root_len == 0, axis=0) == True),
                1,
            )
            self.juvenile.stem_num = np.delete(
                self.juvenile.stem_num,
                np.where(np.all(self.juvenile.stem_num == 0, axis=0) == True),
                1,
            )
            self.juvenile.cover = self.juvenile.veg_frac.sum(axis=1).reshape(-1, 1)

        if self.species == "Salicornia" and self.juvenile.winter == True:
            self.juvenile.veg_frac = np.zeros(_reshape.space)
            self.juvenile.veg_frac = self.juvenile.veg_frac.reshape(
                len(self.juvenile.veg_frac), 1
            )
            self.juvenile.veg_height = np.zeros(self.juvenile.veg_frac.shape)
            self.juvenile.stem_dia = np.zeros(self.juvenile.veg_frac.shape)
            self.juvenile.root_len = np.zeros(self.juvenile.veg_frac.shape)
            self.juvenile.stem_num = np.zeros(self.juvenile.veg_frac.shape)
            self.juvenile.cover = np.zeros(self.juvenile.veg_frac.shape)
            self.juvenile.veg_age = np.zeros(self.juvenile.veg_frac.shape)

        elif np.any(self.juvenile.veg_age > (self.constants.maxYears_LS[0] * 365)):
            self.mature.veg_frac = np.column_stack(
                (self.juvenile.veg_frac[:, -1], self.mature.veg_frac)
            )
            self.mature.veg_height = np.column_stack(
                (self.juvenile.veg_height[:, -1], self.mature.veg_height)
            )
            self.mature.stem_dia = np.column_stack(
                (self.juvenile.stem_dia[:, -1], self.mature.stem_dia)
            )
            self.mature.root_len = np.column_stack(
                (self.juvenile.root_len[:, -1], self.mature.root_len)
            )
            self.mature.stem_num = np.column_stack(
                (self.juvenile.stem_num[:, -1], self.mature.stem_num)
            )
            self.mature.veg_age = np.column_stack(
                (self.juvenile.veg_age[:, -1], self.mature.veg_age)
            )

            self.juvenile.veg_frac = np.delete(self.juvenile.veg_frac, -1, 1)
            self.juvenile.veg_height = np.delete(self.juvenile.veg_height, -1, 1)
            self.juvenile.stem_dia = np.delete(self.juvenile.stem_dia, -1, 1)
            self.juvenile.root_len = np.delete(self.juvenile.root_len, -1, 1)
            self.juvenile.stem_num = np.delete(self.juvenile.stem_num, -1, 1)
            self.juvenile.veg_age = np.delete(self.juvenile.veg_age, -1, 1)
            self.juvenile.cover = self.juvenile.veg_frac.sum(axis=1).reshape(-1, 1)
            self.mature.veg_frac = np.delete(
                self.mature.veg_frac,
                np.where(np.all(self.mature.veg_frac == 0, axis=0) == True),
                1,
            )
            self.mature.veg_height = np.delete(
                self.mature.veg_height,
                np.where(np.all(self.mature.veg_height == 0, axis=0) == True),
                1,
            )
            self.mature.veg_age = np.delete(
                self.mature.veg_age,
                np.where(np.all(self.mature.stem_dia == 0, axis=0) == True),
                1,
            )
            self.mature.stem_dia = np.delete(
                self.mature.stem_dia,
                np.where(np.all(self.mature.stem_dia == 0, axis=0) == True),
                1,
            )
            self.mature.root_len = np.delete(
                self.mature.root_len,
                np.where(np.all(self.mature.root_len == 0, axis=0) == True),
                1,
            )
            self.mature.stem_num = np.delete(
                self.mature.stem_num,
                np.where(np.all(self.mature.stem_num == 0, axis=0) == True),
                1,
            )
            self.mature.cover = self.mature.veg_frac.sum(axis=1).reshape(-1, 1)

        elif np.any(self.juvenile.veg_age > (self.constants.maxAge * 365)):
            self.juvenile.veg_frac = np.delete(self.juvenile.veg_frac, -1, 1)
            self.juvenile.veg_height = np.delete(self.juvenile.veg_height, -1, 1)
            self.juvenile.stem_dia = np.delete(self.juvenile.stem_dia, -1, 1)
            self.juvenile.root_len = np.delete(self.juvenile.root_len, -1, 1)
            self.juvenile.stem_num = np.delete(self.juvenile.stem_num, -1, 1)
            self.juvenile.veg_age = np.delete(self.juvenile.veg_age, -1, 1)
            self.juvenile.cover = self.juvenile.veg_frac.sum(axis=1).reshape(-1, 1)

        if np.any(self.mature.veg_age > (self.constants.maxAge * 365)):
            self.mature.veg_frac = np.delete(self.mature.veg_frac, -1, 1)
            self.mature.veg_height = np.delete(self.mature.veg_height, -1, 1)
            self.mature.stem_dia = np.delete(self.mature.stem_dia, -1, 1)
            self.mature.root_len = np.delete(self.mature.root_len, -1, 1)
            self.mature.stem_num = np.delete(self.mature.stem_num, -1, 1)
            self.mature.veg_age = np.delete(self.mature.veg_age, -1, 1)
            self.mature.cover = self.mature.veg_frac.sum(axis=1).reshape(-1, 1)

            # what if only 1 life stage (e.g. salicornia?)

# Adding up the juvenile and mature fractions of mortality causes

    def total_mort_flood(self, mort: Veg_Mortality):
        return mort.fraction_dead_flood_j + mort.fraction_dead_flood_m

    def total_mort_des(self, mort: Veg_Mortality):
        return mort.fraction_dead_des_j + mort.fraction_dead_des_m

    def total_mort_upr(self, mort: Veg_Mortality):
        return mort.fraction_dead_upr_j + mort.fraction_dead_upr_m

    def total_mort_bursco(self, mort: Veg_Mortality):
        return mort.burial_scour_j + mort.burial_scour_m