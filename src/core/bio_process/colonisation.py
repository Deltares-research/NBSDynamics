import numpy as np
from scipy.optimize import newton
from src.core import RESHAPE
from src.core.base_model import ExtraModel
from src.core.common.constants_veg import Constants
from src.core.vegetation.veg_model import Vegetation
from src.core.common.space_time import VegOnly, DataReshape
from src.core.vegetation.veg_only import VegOnly
from src.core.hydrodynamics.delft3d import FlowFmModel

#Colonisation method (1 = on bare substrate between max and min water levels, 2 = on bare substrate with mud content

#Matlab files: colonization, ColonisationStrategyV1sand, ColonisationStrategyV2mud

#Colonization depends on ColMethod (constant)
# 1. inundation (max, min water level, flooded only in max waterlevel: intertidal area)
# 2. mud fraction in top layer: mud_frac>mud_colonization

#In time: WILL BE DEFINED IN THE SIMULATION FILE (THIS METHODS ONLY CALLED WHEN RIGHT ETS)
# 1.2. amount of ets seed dispersal
# 2.1 Ecotimestep at which colonisation starts
# 2.2 Ecotimestep of last colonisation
#LocEco


class Colonization(ExtraModel):
    """Colonization """
    h: Optional[np.array] = None #Water depth
    cir: Optional[np.ndarray] = None
    ma: Optional[np.ndarray] = None
    constants: Constants = Constants()

    def update(self, veg:Vegetation):
        """Update marsh cover after colonization (settlement)

        :param veg: vegetation
        :type vegetation: Vegetation
        """


    def settlement(self, veg:Vegetation):
        """new vegetation settlement

        :param veg: vegetation
        :type vegetation: Vegetation
        """


    def colonization_criterion(self, veg: Vegetation):
        """determine areas which are available for colonization

        :param veg: vegetation

        :type veg: Vegetation
        """
        if Constants.ColMethod == 1
            self.colonization_inundation_range(veg)
            return self.cir
        elif Constants.ColMethod == 2
            self.colonization_inundation_range(veg)
            self.mud_availability(veg)
            return self.cir and self.ma

    def colonization_inundation_range(self, veg: Vegetation):
        """ Colonization Inundation range
        Args:
            veg (Vegetation): Vegetation
        """
        # # check input
        # if not hasattr(coral.um, "__iter__"):
        #     coral.um = np.array([coral.um])
        # if isinstance(coral.um, (list, tuple)):
        #     coral.um = np.array(coral.um)

        # # Calculations
        self.cir = np.ones(coral.max_wl.shape)
        self.cir = (self.cir_formula(coral.max_wl, coral.min_wl) == 1)
    @staticmethod
    def cir_formula(max_water_level, min_water_level)
        """Determine the Dislodgement Mechanical Threshold.

        :param flow_velocity: depth-averaged flow velocity
        :type flow_velocity: float, numpy.ndarray
        """
        return (max_water_level[max_water_level>0] = 1) - (min_water_level[min_water_depth>0]= 1)

    # matrices for minimum and maximum water levels (water depth (h))

    # look for cells that are flooded during high anf low water levels
    # find seedling location in cells that have water depth only at max. water level
    # for random establishment extract random selection of seedling locations

