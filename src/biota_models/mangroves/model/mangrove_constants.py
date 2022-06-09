import json
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import root_validator

from src.core.common.base_constants import BaseConstants

class MangroveConstants(BaseConstants):
    """Object containing all constants used in marsh_model simulations."""

    species: str

    input_file: Optional[Path]

    warn_proc: bool = False
    # User - Define time - scales
    t_eco_year: int = 24  # number ecological time - steps per year(meaning couplings)

    sim_duration: float = 30  # number of morphological years of entire simulation
    start_date: str = "2022-01-01"  # Start date of the simulation
    winter_days: float = list()

    f: float = 0.3 # Constant of roots number increase % Barend: 0.3(40 cm stem); Danghan: 0.1(~1 m stem and 0.5(18 cm stem)
    TauThres: float = 0.2 # Bed shear stress Threshold for mangrove colonization
    SedThres: float =  0.01 # sedimentation threshold for ColonisationStrategy 2B ( in m) - defined in veg.txt-file
    fl_dr: float = 0.1 # Boundary for water flooding / drying(m)
    t_eco_year: float = 12 # number of ecological time - steps(ets) per year
    num0: float = 750 # initial individuals of plants in one cell
    num_all: float = 2e6 # The max number of columns in one cell incl.plants and roots
    S_cell: int = 2500 # Cell size area
    Mort_plant: float = 10 # Number of plants need to be removed at one time
    Grow_plant: float = 10 # Number of plants need to be grown at one time
    ini_dia: float = 1 #initial stem diameter [cm]


    #Inundation stress factors
    a: float = -8
    b: float = 4
    c: float = 0.5
    d: float = -0.0002 # Competition stress factor coefficient
    ind_a: float = 2.11 # Biomass above-ground constant
    bio_a: float = 0.308 # Biomass above-ground constant
    ind_b: float = 1.17 # Biomass below-ground constant
    bio_b: float = 1.28 # Biomass below-ground constant

    ## TODO B0.5 is dependent on grid cell size! Create formula!
    B_05 = 2.589410**4 # [kg]

    # growth parameter
    G: float = 254 # Growth constant [cm]
    b2: float = 80 # Growth constant [-]
    b3: float = 1  # Growth constant [-]
    MaxH: float = 1600 # maximum tree height [cm]
    MaxD: float = 40 # maximum stem diameter [cm]

    # root information
    m: float = 1000 # maximum number of roots per tree
    f: float = 0.3 #constant describing rate of increase of roots
