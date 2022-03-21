from pathlib import Path
from typing import Optional
import numpy as np
from pydantic import root_validator, validator
from src.core.base_model import BaseModel

class Constants(BaseModel):
    """Object containing all constants used in marsh_model simulations."""

    #Input file
    input_file: Optional[Path]

    #Processes
    warn_proc: bool = False

    # User - Define time - scales
    t_eco_year: float = 24 # number ecological time - steps per year(meaning couplings)
    years: float = 30 # number of morphological years of entire simulation(total simulation time * morfac / hydrodynamic time that defines one year)

    #Colonization
    ColMethod: int = 1 # Colonisation method (1 = on bare substrate between max and min water levels, 2 = on bare substrate with mud content)
    #for now only the method 1 is implemented!, mud content in top layer needed from DFM
    etsSeed: int = 2 # Amount of ets seed dispersal = 1
    ColStart: int = 7 # Ecotimestep at which colonisation starts
    ColEnd: int = 8 # Ecotimestep of last colonisation
    random: int = 5 # random colonization as described in Bij de Vaate et al., 2020 with n number of cells colonized as fraction: n = SeedlingLocations/random
    mud_colonization: tuple = (0.0, 0.0) # 3.8. mud percentage for colonization dependent on life stage

    fl_dr: float = 0.05 # Boundary for flooding/drying threshold used in the vegetation computations [m]

    # Vegetation
    # GENERAL CHARACTERISTICS
    maxAge: int = 20 # 1.1. maximum age
    vegFormula: int = 154 # 1.5. vegetation formula (is fixed with Baptiste 2007: 154, for different formula other parameters are required)
    num_ls: int = 2 # 1.6. number of life stages
    iniRoot: float = 0.05 # 1.7. initial root length in m
    iniShoot: float = 0.015 # 1.8. initial shoot length in m
    iniDia: float = 0.003 # 1.9. initial stem diameter in m
    vegType: int = 1 #Vegetation type (1 = riparian, 2 = aquatic, 3 = amphibic)
    etsShoot_start: int = 7 # 1.14 Ecological timestep start growth shoot
    etsShoot_end: int = 19# 1.15 Ecological timestep end growth shoot
    etsWinter_start: int = 21 # 1.16 Ecotime step start winter period

    # LIFE STAGE CHARACTERISTICS (dependent on nls above): here 2
    maxGrowth_H: float = (0.8, 1.3) # 3.1. maximum plant height growth [m]
    maxDia: float = (0.003, 0.005) # 3.2. max stem diameter at ets "end growth shoot"
    maxRoot: float = (0.2, 1) # 3.3. max root length [m] at ets "end growth root"
    maxYears_LS: float = (1, 19) # 3.4. maximum number of years in lifestage
    numStem: float = (700, 700) # 3.5. number of stems per m2
    iniCol_frac: float = (0.6, 0.6) # 3.6. initial colonization fraction (0-1)
    Cd: float = (1.1, 1.15) # 3.7. drag coefficient
    desMort_thres: float = (400, 400) # 3.9. dessication mortality threshold
    desMort_slope: float = (0.75, 0.75) # 3.10. dessication mortality slope
    floMort_thres: float = (0.4, 0.4) # 3.11. flooding mortality threshold
    floMort_slope: float = (0.25, 0.25)# 3.12. flooding mortality slope
    vel_thres: float = (0.15, 0.25)   # 3.13. flow velocity threshold
    vel_slope: float = (3, 3)  # 3.14. flow velocity slope
    maxH_winter: float = (0.4, 0.4)  # 3.15  max height during winter time
    # 3.16  Salinity tolerance placeholder (excluded here)

    @classmethod
    def from_input_file(cls, input_file: Path):
        """
        Generates a 'Constants' class based on the defined parameters in the input_file.

        Args:
            input_file (Path): Path to the constants input (.txt) file.
        """

        def split_line(line: str):
            s_line = line.split("=")
            if len(s_line) <= 1:
                raise ValueError
            return s_line[0].strip(), s_line[1].strip()

        def format_line(line: str) -> str:
            return split_line(line.split("#")[0])

        def normalize_line(line: str) -> str:
            return line.strip()

        input_lines = [
            format_line(n_line)
            for line in input_file.read_text().splitlines(keepends=False)
            if line and not (n_line := normalize_line(line)).startswith("#")
        ]
        cls_constants = cls(**dict(input_lines))
        #cls_constants.correct_values()
        return cls_constants