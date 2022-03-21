from typing import Dict, Optional, Union

import numpy as np
from pydantic import validator

from src.core.common.singletons import RESHAPE
from src.core.base_model import ExtraModel
from src.core.common.constants_veg import Constants
from src.core.common.space_time import DataReshape
from src.core.vegetation.veg_only import VegOnly

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
    veg_age: VegAttribute #vegetation age [yrs]
    stem_num: VegAttribute #number of stems
    ets: VegAttribute #current state of ets, increases over the year (1:constant.ets_per_year)

    # other attributes.
    _cover: Optional[VegAttribute] = None
    # m_height: Optional[VegAttribute] = None
    # m_root: Optional[VegAttribute] = None
    # m_stemdia: Optional[VegAttribute] = None

    # # flow micro environment
    # ucm: Optional[VegAttribute] = None
    # um: Optional[VegAttribute] = None
    # delta_t: Optional[VegAttribute] = None



    @validator("veg_height", "stem_dia", "veg_den", "root_len", "veg_ls", "veg_age", "stem_num")
    @classmethod
    def validate_vegetation_attribute(
       cls, value: Optional[VegAttribute]
    ) -> object | None:
       if value is None:
           return value
       return DataReshape.variable2array(value)

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
    def veg_den(self):
        """density (stem diameter * number of stems) for Baptist formula"""
        return self.stem_dia * self.stem_num



    @property
    def slope_height(self):
        self.m_height =  (Constants.maxGrowth_H[self.veg_ls] - self.veg_height)/ (Constants.etsShoot_end- Constants.etsShoot_end) #slope: meters per ets that plant grows
        return self.m_height

    @property
    def slope_root(self):
        self.m_root = (Constants.maxRoot[self.veg_ls]- self.veg_root)/  (Constants.etsShoot_end- Constants.etsShoot_end)
        return self.m_root

    @property
    def slope_stemdia(self):
        self.m_stemdia =  (Constants.maxDia[self.veg_ls] - self.stem_dia)/ (Constants.etsShoot_end- Constants.etsShoot_end) #slope: meters per ets that plant grows
        return self.m_stemdia

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
    def age_matrix(self):
        """self.RESHAPEd vegetation age."""
        return RESHAPE().variable2matrix(self.veg_age, "space")

    @property
    def stemNum_matrix(self):
        """self.RESHAPEd vegetation age."""
        return RESHAPE().variable2matrix(self.stem_num, "space")

    @property
    def cover(self):
        """Carrying capacity."""
        if self._cover is None:
            cover = np.ones(np.array(self.veg_den).shape)
            cover[self.veg_den == 0.0] = 0.0  # 21.09 made 0. instead of just zero
            return cover

        return self._cover

    def update_vegetation_characteristics(self, ):
    """
    update vegetation characterteristics based on
    the vegetations age (in ets) and fraction of veg in each cell

    """

    # update_density
    # update_cover
    #def update_lifestage(self):
        #will get values 0 and 1, change to 1 in second year
    #update age
    #update fraction
    #update drag coefficient
    # ...

#USE THIS FUNTION IN THE MORTAILITY, GROWTH & Settlement
# to determine new vegetation characteristics!
    # add the new vegetation coming due to colonization
    # remove vegetation which dies due to mortality


    #def update_lifestage(self):
        #will get values 0 and 1
    #change to 1 in second year


