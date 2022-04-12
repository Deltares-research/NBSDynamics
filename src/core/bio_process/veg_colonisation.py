from typing import Optional
import numpy as np
from src.core.base_model import ExtraModel
from src.core.common.constants_veg import Constants
from src.core.vegetation.veg_model import Vegetation
from src.core.hydrodynamics.delft3d import FlowFmModel


class Colonization(ExtraModel):
    """
    Colonization
    Colonization depends on ColMethod (Colonisation method (1 = on bare substrate between max and min water levels, 2 = on bare substrate with mud content)
    1. inundation (max, min water level, flooded only in max waterlevel: intertidal area)
    2. mud fraction in top layer: mud_frac>mud_colonization (NOT YET FULLY IMPLEMENTED!)
    """

    cir: Optional[np.ndarray] = None
    ma: Optional[np.ndarray] = None

    def __init__(self):
        super().__init__()
        self.seed_loc = None

    def update(self, veg: Vegetation, constants):
        """Update marsh cover after colonization (settlement)
        ONLY IF WE ARE IN THE RIGHT ETS!

        :param veg: vegetation
        :type vegetation: Vegetation
        """

        # # available locations for settlement
        Colonization.col_location(self, veg, constants)
        # TODO check this!
        loc = veg.initial.veg_frac[self.seed_loc]
        loc[veg.total_cover[self.seed_loc] <= (1 - constants.iniCol_frac)] = 1
        veg.initial.veg_frac[self.seed_loc] = loc*constants.iniCol_frac
        veg.initial.veg_height[self.seed_loc] = loc*constants.iniShoot
        veg.initial.stem_dia[self.seed_loc] = loc*constants.iniDia
        veg.initial.root_len[self.seed_loc] = loc*constants.iniRoot
        veg.initial.stem_num[self.seed_loc] = loc*constants.num_stem[0]


    def col_location(self, veg, constants):
        """new vegetation settlement

        :param veg: vegetation
        :type vegetation: Vegetation
        """
        # find seedling location in cells that have water depth only at max. water level
        # for random establishment extract random selection of seedling locations
        self.seed_loc = np.where(self.colonization_criterion(veg, constants) == True)  # all possible locations for seedlings
        if constants.random == 0:
            self.seed_loc = self.seed_loc[0]
        else:
            self.seed_loc = np.random.choice(self.seed_loc[0], round(
                len(self.seed_loc[0]) / constants.random))  # locations where random settlement can occur

    def colonization_criterion(self, veg: Vegetation, constants):
        """determine areas which are available for colonization

        :param veg: vegetation

        :type veg: Vegetation
        """
        # if self.constants.ColMethod == 1:
        self.colonization_inundation_range(veg)
        return self.cir
        # elif self.constants.ColMethod == 2:
        #     self.colonization_inundation_range(veg)
        #     self.mud_availability(veg, constants)
        #     return np.logical_and(self.cir, self.ma) #matrix with true everywhere where vegetation is possible according to mud content and inundation

    def colonization_inundation_range(self, veg: Vegetation):
        """ Colonization Inundation range
        Args:
            veg (Vegetation): Vegetation
        """

        # # Calculations
        self.cir = np.zeros(veg.max_wl.shape)
        self.cir = (self.cir_formula(veg.max_wl,
                                     veg.min_wl) == 1)  # true, false matrix look for cells that are flooded during high anf low water levels

    @staticmethod
    def cir_formula(max_water_level, min_water_level):
        max_water_level[max_water_level > 0] = 1
        min_water_level[min_water_level > 0] = 1
        return max_water_level - min_water_level

##TODO get information on mud in top layer from DFM

# def mud_availability(self, veg: Vegetation, constants):
#     """ Colonization criterion for mud availability
#            Args:
#                veg (Vegetation): Vegetation
#     """
#     self.ma = np.ones(FlowFmModel.space.shape)
#     self.ma = veg.mud_fract > constants.mud_colonization(veg.veg_ls) #matrix with false and true
