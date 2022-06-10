from typing import Optional

import numpy as np

from src.biota_models.vegetation.model.veg_constants import VegetationConstants
from src.biota_models.vegetation.model.veg_model import Vegetation
from src.core.base_model import ExtraModel


class Colonization(ExtraModel):
    """
    Colonization
    Colonization depends on ColMethod (Colonisation method (1 = on bare substrate between max and min water levels, 2 = on bare substrate with mud content)
    1. inundation (max, min water level, flooded only in max waterlevel: intertidal area)
    2. mud fraction in top layer: mud_frac>mud_colonization (NOT YET FULLY IMPLEMENTED!)
    """

    cir: Optional[np.ndarray]
    ma: Optional[np.ndarray]
    seed_loc: Optional[np.ndarray]
    seed_loc1: Optional[np.ndarray]
    seed_loc2: Optional[np.ndarray]

    def update(
        self, veg_species1: Vegetation, veg_species2: Optional[Vegetation] = None, veg_species3: Optional[Vegetation] = None
    ):
        """Update marsh cover after colonization (settlement)
        if two vegetation objects are given (different species),
        they will compete for space when they colonize

        :param veg: vegetation

        :type vegetation: Vegetation
        """
        if not veg_species2 and not veg_species3:
            # # available locations for settlement
            Colonization.col_location(self, veg_species1)
            loc = veg_species1.initial.veg_frac[self.seed_loc]
            loc[
                veg_species1.total_cover[self.seed_loc]
                <= (1 - veg_species1.constants.iniCol_frac)
            ] = 1
            veg_species1.initial.veg_frac[self.seed_loc] = (
                loc * veg_species1.constants.iniCol_frac
            )
            veg_species1.initial.veg_height[self.seed_loc] = (
                loc * veg_species1.constants.iniShoot
            )
            veg_species1.initial.stem_dia[self.seed_loc] = (
                loc * veg_species1.constants.iniDia
            )
            veg_species1.initial.root_len[self.seed_loc] = (
                loc * veg_species1.constants.iniRoot
            )
            veg_species1.initial.stem_num[self.seed_loc] = (
                loc * veg_species1.constants.num_stem[0]
            )

        # TODO test this!
        elif not veg_species3:
            total_cover = veg_species1.total_cover + veg_species2.total_cover

            Colonization.col_location(self, veg_species1)
            self.seed_loc1 = self.seed_loc
            Colonization.col_location(self, veg_species2)
            self.seed_loc2 = self.seed_loc

            loc1 = veg_species1.initial.veg_frac[self.seed_loc1]
            loc1[
                total_cover[self.seed_loc1] <= (1 - veg_species1.constants.iniCol_frac)
            ] = 1
            loc2 = veg_species2.initial.veg_frac[self.seed_loc2]
            loc2[
                total_cover[self.seed_loc2] <= (1 - veg_species2.constants.iniCol_frac)
            ] = 1

            veg_species1.initial.veg_height[self.seed_loc1] = (
                loc1 * veg_species1.constants.iniShoot
            )
            veg_species1.initial.stem_dia[self.seed_loc1] = (
                loc1 * veg_species1.constants.iniDia
            )
            veg_species1.initial.root_len[self.seed_loc1] = (
                loc1 * veg_species1.constants.iniRoot
            )
            veg_species1.initial.stem_num[self.seed_loc1] = (
                loc1 * veg_species1.constants.num_stem[0]
            )

            veg_species2.initial.veg_height[self.seed_loc2] = (
                loc2 * veg_species2.constants.iniShoot
            )
            veg_species2.initial.stem_dia[self.seed_loc2] = (
                loc2 * veg_species2.constants.iniDia
            )
            veg_species2.initial.root_len[self.seed_loc2] = (
                loc2 * veg_species2.constants.iniRoot
            )
            veg_species2.initial.stem_num[self.seed_loc2] = (
                loc2 * veg_species2.constants.num_stem[0]
            )

            # comp = np.where(loc1 == 1 and loc2 == 1)
            if (
                veg_species1.constants.iniCol_frac + veg_species2.constants.iniCol_frac
                > 1
            ):
                loc1[np.in1d(self.seed_loc1, self.seed_loc2) == True] = 1 / (
                    veg_species1.constants.iniCol_frac
                    + veg_species2.constants.iniCol_frac
                )
                loc2[np.in1d(self.seed_loc2, self.seed_loc1) == True] = 1 / (
                    veg_species1.constants.iniCol_frac
                    + veg_species2.constants.iniCol_frac
                )

            loc1[loc1 > 1] = 1
            loc2[loc2 > 1] = 1

            veg_species1.initial.veg_frac[self.seed_loc1] = (
                loc1 * veg_species1.constants.iniCol_frac
            )
            veg_species2.initial.veg_frac[self.seed_loc2] = (
                loc2 * veg_species2.constants.iniCol_frac
            )

        else:
            total_cover = veg_species1.total_cover + veg_species2.total_cover + veg_species3.total_cover

            Colonization.col_location(self, veg_species1)
            self.seed_loc1 = self.seed_loc
            Colonization.col_location(self, veg_species2)
            self.seed_loc2 = self.seed_loc
            Colonization.col_location(self, veg_species3)
            self.seed_loc3 = self.seed_loc

            loc1 = veg_species1.initial.veg_frac[self.seed_loc1]
            loc1[
                total_cover[self.seed_loc1] <= (1 - veg_species1.constants.iniCol_frac)
                ] = 1
            loc2 = veg_species2.initial.veg_frac[self.seed_loc2]
            loc2[
                total_cover[self.seed_loc2] <= (1 - veg_species2.constants.iniCol_frac)
                ] = 1
            loc3 = veg_species3.initial.veg_frac[self.seed_loc3]
            loc3[
                total_cover[self.seed_loc3] <= (1 - veg_species3.constants.iniCol_frac)
                ] = 1

            veg_species1.initial.veg_height[self.seed_loc1] = (
                    loc1 * veg_species1.constants.iniShoot
            )
            veg_species1.initial.stem_dia[self.seed_loc1] = (
                    loc1 * veg_species1.constants.iniDia
            )
            veg_species1.initial.root_len[self.seed_loc1] = (
                    loc1 * veg_species1.constants.iniRoot
            )
            veg_species1.initial.stem_num[self.seed_loc1] = (
                    loc1 * veg_species1.constants.num_stem[0]
            )

            veg_species2.initial.veg_height[self.seed_loc2] = (
                    loc2 * veg_species2.constants.iniShoot
            )
            veg_species2.initial.stem_dia[self.seed_loc2] = (
                    loc2 * veg_species2.constants.iniDia
            )
            veg_species2.initial.root_len[self.seed_loc2] = (
                    loc2 * veg_species2.constants.iniRoot
            )
            veg_species2.initial.stem_num[self.seed_loc2] = (
                    loc2 * veg_species2.constants.num_stem[0]
            )

            veg_species3.initial.veg_height[self.seed_loc3] = (
                    loc3 * veg_species3.constants.iniShoot
            )
            veg_species3.initial.stem_dia[self.seed_loc3] = (
                    loc3 * veg_species3.constants.iniDia
            )
            veg_species3.initial.root_len[self.seed_loc3] = (
                    loc3 * veg_species3.constants.iniRoot
            )
            veg_species3.initial.stem_num[self.seed_loc3] = (
                    loc3 * veg_species3.constants.num_stem[0]
            )


            if (
                    veg_species1.constants.iniCol_frac + veg_species2.constants.iniCol_frac + veg_species3.constants.iniCol_frac
                    > 1
            ):
                a = 1
                ## TODO How to do this??
                # 1 with 2
                loc1[np.in1d(self.seed_loc1, self.seed_loc2) == True] = 1 / (
                        veg_species1.constants.iniCol_frac
                        + veg_species2.constants.iniCol_frac
                )
                loc2[np.in1d(self.seed_loc2, self.seed_loc1) == True] = 1 / (
                        veg_species1.constants.iniCol_frac
                        + veg_species2.constants.iniCol_frac
                )
                # 1 with 3
                loc1[np.in1d(self.seed_loc1, self.seed_loc3) == True] = 1 / (
                        veg_species1.constants.iniCol_frac
                        + veg_species3.constants.iniCol_frac
                )
                loc3[np.in1d(self.seed_loc3, self.seed_loc1) == True] = 1 / (
                        veg_species1.constants.iniCol_frac
                        + veg_species3.constants.iniCol_frac
                )
                # 2 with 3
                loc2[np.in1d(self.seed_loc2, self.seed_loc3) == True] = 1 / (
                        veg_species2.constants.iniCol_frac
                        + veg_species3.constants.iniCol_frac
                )
                loc3[np.in1d(self.seed_loc3, self.seed_loc2) == True] = 1 / (
                        veg_species2.constants.iniCol_frac
                        + veg_species3.constants.iniCol_frac
                )
                loc1[np.logical_and(np.in1d(self.seed_loc1, self.seed_loc2) == True, np.in1d(self.seed_loc1, self.seed_loc3) == True)] = 1 / (
                        veg_species1.constants.iniCol_frac
                        + veg_species2.constants.iniCol_frac
                        + veg_species3.constants.iniCol_frac
                )
                loc2[np.logical_and(np.in1d(self.seed_loc2, self.seed_loc1) == True, np.in1d(self.seed_loc2, self.seed_loc3) == True)] = 1 / (
                        veg_species1.constants.iniCol_frac
                        + veg_species2.constants.iniCol_frac
                        + veg_species3.constants.iniCol_frac
                )
                loc3[np.logical_and(np.in1d(self.seed_loc3, self.seed_loc1) == True, np.in1d(self.seed_loc3, self.seed_loc2) == True)] = 1 / (
                        veg_species1.constants.iniCol_frac
                        + veg_species2.constants.iniCol_frac
                        + veg_species3.constants.iniCol_frac
                )

                # all three can colonize
                # loc12 = np.in1d(self.seed_loc3, self.seed_loc2)
                # loc13 =
                # loc123 = np.in1d(loc12, self.seed_loc3)
                #
                # loc1[loc123] = 1 / (
                #         veg_species2.constants.iniCol_frac
                #         + veg_species2.constants.iniCol_frac
                #         + veg_species3.constants.iniCol_frac
                # )
                #
                # loc2[loc123] = 1 / (
                #         veg_species1.constants.iniCol_frac
                #         + veg_species2.constants.iniCol_frac
                #         + veg_species3.constants.iniCol_frac
                # )
                # loc3[loc123] = 1 / (
                #         veg_species2.constants.iniCol_frac
                #         + veg_species2.constants.iniCol_frac
                #         + veg_species3.constants.iniCol_frac
                # )
            loc1[loc1 > 1] = 1
            loc2[loc2 > 1] = 1
            loc3[loc3 > 1] = 1
            veg_species1.initial.veg_frac[self.seed_loc1] = (
                    loc1 * veg_species1.constants.iniCol_frac
            )
            veg_species2.initial.veg_frac[self.seed_loc2] = (
                    loc2 * veg_species2.constants.iniCol_frac
            )
            veg_species3.initial.veg_frac[self.seed_loc3] = (
                    loc3 * veg_species3.constants.iniCol_frac
            )


    def col_location(self, veg: Vegetation):
        """
        new vegetation settlement

        :param veg: vegetation
        :type vegetation: Vegetation
        """
        # find seedling location in cells that have water depth only at max. water level
        # for random establishment extract random selection of seedling locations
        self.seed_loc = np.where(
            Colonization.colonization_criterion(self, veg) == True
        )  # all possible locations for seedlings
        if veg.constants.random == 0:
            self.seed_loc = self.seed_loc[0]
        else:
            self.seed_loc = np.random.choice(
                self.seed_loc[0], round(len(self.seed_loc[0]) * (veg.constants.random/100))
            )  # locations where random settlement can occur considering the propability of establishment (constants.random [%])

    def colonization_criterion(self, veg: Vegetation):
        """determine areas which are available for colonization

        :param veg: vegetation

        :type veg: Vegetation
        """
        # if veg.constants.ColMethod == 1:
        Colonization.colonization_inundation_range(self, veg)
        return self.cir
        # elif self.constants.ColMethod == 2:
        #     Colonization.colonization_inundation_range(self, veg)
        #     Colonization.mud_availability(veg)
        #     return np.logical_and(self.cir, self.ma) #matrix with true everywhere where vegetation is possible according to mud content and inundation

    def colonization_inundation_range(self, veg: Vegetation):
        """Colonization Inundation range
        Args:
            veg (Vegetation): Vegetation
        """

        # # Calculations
        self.cir = np.zeros(veg.max_wl.shape)
        self.cir = (
            Colonization.cir_formula(veg.max_wl, veg.min_wl) == 1
        )  # true, false matrix look for cells that are flooded during high anf low water levels

    @staticmethod
    def cir_formula(max_water_level, min_water_level):
        max_water_level[max_water_level > 0] = 1
        min_water_level[min_water_level > 0] = 1
        return max_water_level - min_water_level


##TODO get information on mud in top layer from DFM

# def mud_availability(self, veg: Vegetation):
#     """ Colonization criterion for mud availability
#            Args:
#                veg (Vegetation): Vegetation
#     """
#     self.ma = np.ones(FlowFmModel.space.shape)
#     self.ma = veg.mud_fract > veg.constants.mud_colonization(veg.veg_ls) #matrix with false and true
