from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import newton
from src.core import RESHAPE
from src.core.base_model import ExtraModel
from src.core.common.constants_veg import Constants
from src.core.vegetation.veg_model import Vegetation
from random import sample
from src.core.common.space_time import VegOnly, DataReshape
from src.core.vegetation.veg_only import VegOnly
from src.core.hydrodynamics.delft3d import FlowFmModel

class Veg_Mortaility(ExtraModel):
    """Mortality"""

    drown: Optional[np.array] = None
    uproot: Optional[np.array] = None
    burial_scour: Optional[np.array] = None
    loc_mortality: Optional[np.array] = None
    constants: Constants = Constants()

    def update(self, veg:Vegetation, constants):
        """Update vegetation characteristics after mortality"""


    def mort_loc(self, veg:Vegetation, constants):
        """locations where the vegetation dies"""
        ##TODO see if this makes sense once I know whats in the arrays
        self.loc_mortality = self.drown + self.uproot + self.burial_scour

    def mort_criterion(self, veg:Vegetation, constants):
        """criterion for vegetation mortality"""
        # drowning by hydroperiod (flooding thresholds)
        self.drown = np.ones(FlowFmModel.space.shape)
        self.drown = self.drowing_hydroperiod(veg, constants)
        # uprooting due to high tidal flow velocities (flow velocity thresholds)
        self.uproot = np.ones(FlowFmModel.space.shape)
        self.uproot = self.uprooting(veg, constants)
        # burial and scour through erosion and sedimentation (dessication thresholds)
        self.burial_scour = np.ones(FlowFmModel.space.shape)
        self.burial_scour = self.erosion_sedimentation(veg, constants)

    def drowning_hydroperiod(self, veg: Vegetation, constants):
        return # array with cells where vegetation dies due to HP

    def uprooting(self, veg: Vegetation, constants):
        return # array with cells where vegetation dies due to uprooting

    def erosion_sedimentation(self, veg: Vegetation, constants):
        return # array with cells where vegetation dies due to burial and scour
