import numpy as np
from scipy.optimize import newton
from src.core import RESHAPE
from src.core.base_model import ExtraModel
from src.core.common.constants_veg import Constants
from src.core.vegetation.veg_model import Vegetation
from random import sample
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
    seedloc: Optional[np.array] = None
    settlement: Optional[np.array] = None
    constants: Constants = Constants()

    def update(self, veg:Vegetation):
        """Update marsh cover after colonization (settlement)
        ONLY IF WE ARE IN THE RIGHT ETS!

        :param veg: vegetation
        :type vegetation: Vegetation
        """
        if veg.ets in range(Constants.ColStart, Constants.ColEnd, 1):
        # # available locations for settlement
            Colonization.settlement(self, veg, constants)
        # TODO find if space is available!
        # # update IF SPACE AVAILABLE!!! How is vegetation info stored?
        # population states

        # vegetation characteristics
        else:
            pass

    def settlement(self, veg, constants):
        """new vegetation settlement

        :param veg: vegetation
        :type vegetation: Vegetation
        """
        # find seedling location in cells that have water depth only at max. water level
        # for random establishment extract random selection of seedling locations
        self.seedloc = np.where(Colonization.colonization_criterion(veg) ==True) #all possible locations for seedlings
        if constants.random ==0:
            self.settlement = self.seedloc
        else:
            self.settlement = np.random.choice(self.seedloc,round(len(self.seedloc)/constants.random)) #locations where random settlement can occur

    def colonization_criterion(self, veg: Vegetation):
        """determine areas which are available for colonization

        :param veg: vegetation

        :type veg: Vegetation
        """
        if Constants.ColMethod == 1:
            self.colonization_inundation_range(veg)
            return self.cir
        elif Constants.ColMethod == 2:
            self.colonization_inundation_range(veg)
            self.mud_availability(veg)
            return np.logical_and(self.cir, self.ma) #matrix with true everywhere where vegetation is possible according to mud content and inundation

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
        self.cir = np.ones(veg.max_wl.shape)
        self.cir = (self.cir_formula(veg.max_wl, veg.min_wl) == 1)     #true, false matrix look for cells that are flooded during high anf low water levels
    @staticmethod
    def cir_formula(max_water_level, min_water_level)
        """Determine the Dislodgement Mechanical Threshold.

        :param flow_velocity: depth-averaged flow velocity
        :type flow_velocity: float, numpy.ndarray
        """
        return (max_water_level[max_water_level>0] = 1) - (min_water_level[min_water_level >0] = 1)

    def mud_availability(self, veg: Vegetation):
        """ Colonization criterion for mud availability
               Args:
                   veg (Vegetation): Vegetation
        """
        self.ma = np.ones(veg.mud_fract.shape)
        self.ma = veg.mud_fract > Constants.mud_colonization(veg.veg_ls) #matrix with false and true





