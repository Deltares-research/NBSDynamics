from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from pydantic import root_validator, validator
from src.core.base_model import BaseModel
from datetime import datetime
from datetime import timedelta


class Constants(BaseModel):
    """Object containing all constants used in marsh_model simulations."""

    #Input file
    input_file: Optional[Path]

    #Processes
    warn_proc: bool = False

    # User - Define time - scales
    t_eco_year: int = 24 # number ecological time - steps per year(meaning couplings)
    ## TODO check with MorFac, what years is this then?
    sim_duration: float = 30 # number of morphological years of entire simulation
    start_date: str = "2022-01-01" #Start date of the simulation
    #Colonization
    ColMethod: int = 1 # Colonisation method (1 = on bare substrate between max and min water levels, 2 = on bare substrate with mud content)
    ets_per_year: int = 24 #number of ecological time steps per year

    # def __init__(self, species_1='Spartina anglica', species_2=None):
    #
    #     super().__init__()
    #     self.species_1 = species_1
    #     self.species_2 = species_2
    #     self.get_constants(self, self.species_1, self.species_2)
    #
    #     def get_constants(self, self.species_1, self.species_2):
    #         if species_1 is not None and species_2 is None:
    #             with open(fpath_constants_file) as f:
    #                 constants_dict = json.load(f)
    #                 self.ColStart = constants_dict[self.species_1]['ColStart']
    #                 self.ColEnd = constants_dict[self.species_1]['ColEnd']
    #                 self.random = constants_dict[self.species_1]['random']
    #                 self.ColStart = constants_dict[self.species_1]['ColStart']
    #                 self.mud_col = constants_dict[self.species_1]['mud_colonization']
    #                 self.fl_dr = constants_dict[self.species_1]['fl_dr']
    #                 self.maxAge = constants_dict[self.species_1]['Maximum age']
    #                 self.num_ls = constants_dict[self.species_1]['Number LifeStages']
    #                 self.iniRoot = constants_dict[self.species_1]['initial root length']
    #                 self.iniShoot = constants_dict[self.species_1]['initial shoot length']
    #                 self.iniDia = constants_dict[self.species_1]['initial diameter']
    #                 self.growth_start = constants_dict[self.species_1]['start growth period']
    #                 self.growth_end = constants_dict[self.species_1]['end growth period']
    #                 self.winter_start = constants_dict[self.species_1]['start winter period']
    #                 self.maxGrowth_H = constants_dict[self.species_1]['maximum plant height']
    #                 self.maxDia = constants_dict[self.species_1]['maximum diameter']
    #                 self.maxRoot = constants_dict[self.species_1]['maximum root length']
    #                 self.maxYears_LS = constants_dict[self.species_1]['maximum years in LifeStage']
    #                 self.num_stem = constants_dict[self.species_1]['numStem']
    #                 self.iniCol_frac = constants_dict[self.species_1]['iniCol_frac']
    #                 self.Cd = constants_dict[self.species_1]['Cd']
    #                 self.desMort_thres = constants_dict[self.species_1]['desMort_thres']
    #                 self.desMort_slope = constants_dict[self.species_1]['desMort_slope']
    #                 self.floMort_thres = constants_dict[self.species_1]['floMort_thres']
    #                 self.floMort_slope = constants_dict[self.species_1]['floMort_slope']
    #                 self.vel_thres = constants_dict[self.species_1]['vel_thres']
    #                 self.vel_slope = constants_dict[self.species_1]['vel_slope']
    #                 self.maxH_winter = constants_dict[self.species_1]['maxH_winter']
    #
    #         elif self.species_1 is not None and self.species_2 is not None:
    #             species = [self.species_1, self.species_2]
    #             self.ColStart = list()
    #             self.ColEnd = list()
    #             self.random = list()
    #             self.ColStart = list()
    #             self.mud_col = list()
    #             self.fl_dr = list()
    #             self.maxAge = list()
    #             self.num_ls = list()
    #             self.iniRoot = list()
    #             self.iniShoot = list()
    #             self.iniDia = list()
    #             self.growth_start = list()
    #             self.growth_end = list()
    #             self.winter_start = list()
    #             self.maxGrowth_H = list()
    #             self.maxDia = list()
    #             self.maxRoot = list()
    #             self.maxYears_LS = list()
    #             self.num_stem = list()
    #             self.iniCol_frac = list()
    #             self.Cd =list()
    #             self.desMort_thres = list()
    #             self.desMort_slope = list()
    #             self.floMort_thres = list()
    #             self.floMort_slope = list()
    #             self.vel_thres = list()
    #             self.vel_slope = list()
    #             self.maxH_winter = list()
    #             with open(fpath_constants_file) as f:
    #                 constants_dict = json.load(f)
    #                 for specie in species:
    #                     self.ColStart.append(constants_dict[specie]['ColStart'])
    #                     self.ColEnd.append(constants_dict[specie]['ColStart'])
    #                     self.random.append(constants_dict[specie]['ColStart'])
    #                     self.ColStart.append(constants_dict[specie]['ColStart'])
    #                     self.mud_col.append(constants_dict[specie]['ColStart'])
    #                     self.fl_dr.append(constants_dict[specie]['ColStart'])
    #                     self.maxAge.append(constants_dict[specie]['ColStart'])
    #                     self.num_ls.append(constants_dict[specie]['ColStart'])
    #                     self.iniRoot.append(constants_dict[specie]['ColStart'])
    #                     self.iniShoo.append(constants_dict[specie]['ColStart'])
    #                     self.iniDia.append(constants_dict[specie]['ColStart'])
    #                     self.growth_start.append(constants_dict[specie]['ColStart'])
    #                     self.growth_end.append(constants_dict[specie]['ColStart'])
    #                     self.winter_start.append(constants_dict[specie]['ColStart'])
    #                     self.maxGrowth_H.append(constants_dict[specie]['ColStart'])
    #                     self.maxDia.append(constants_dict[specie]['ColStart'])
    #                     self.maxRoot.append(constants_dict[specie]['ColStart'])
    #                     self.maxYears_LS.append(constants_dict[specie]['ColStart'])
    #                     self.num_stem.append(constants_dict[specie]['ColStart'])
    #                     self.iniCol_frac.append(constants_dict[specie]['ColStart'])
    #                     self.Cd.append(constants_dict[specie]['ColStart'])
    #                     self.desMort_thres.append(constants_dict[specie]['ColStart'])
    #                     self.desMort_slope.append(constants_dict[specie]['ColStart'])
    #                     self.floMort_thres.append(constants_dict[specie]['ColStart'])
    #                     self.floMort_slope.append(constants_dict[specie]['ColStart'])
    #                     self.vel_thres.append(constants_dict[specie]['ColStart'])
    #                     self.vel_slope.append(constants_dict[specie]['ColStart'])
    #                     self.maxH_winter.append(constants_dict[specie]['ColStart'])
    #         else:
    #             pass
    #


    ColStart: str = "2022-04-01" # Date at which colonisation starts (year,month,day)
    ColEnd: str = "2022-05-31" # Date of last colonisation (year,month,day)

    random: int = 30 # random colonization as described in Bij de Vaate et al., 2020 with n number of cells colonized as fraction: n = SeedlingLocations/random
    mud_colonization: float = [0.0, 0.0] # 3.8. mud percentage for colonization dependent on life stage

    fl_dr: float = 0.05  # Boundary for flooding/drying threshold used in the vegetation computations [m]

    # Vegetation
    # GENERAL CHARACTERISTICS
    maxAge: int = 20 # 1.1. maximum age
    vegFormula: int = 154 # 1.5. vegetation formula (is fixed with Baptiste 2007: 154, for different formula other parameters are required)
    num_ls: int = 2 # 1.6. number of life stages
    iniRoot: float = 0.05 # 1.7. initial root length in m
    iniShoot: float = 0.015 # 1.8. initial shoot length in m
    iniDia: float = 0.003 # 1.9. initial stem diameter in m

    growth_start: str = "2022-04-01" # 1.14 Date start growth shoot
    growth_end: str = "2022-10-31" # 1.15 Date end growth shoot (year, month, day)
    winter_start: str = "2022-11-30" # 1.16 Date start winter period

    # LIFE STAGE CHARACTERISTICS (dependent on nls above): here 2
    maxGrowth_H: float = [0.8, 1.3] # 3.1. maximum plant height growth [m]
    maxDia: float = [0.003, 0.005] # 3.2. max stem diameter at "end growth shoot"
    maxRoot: float = [0.2, 1] # 3.3. max root length [m] at "end growth root"
    maxYears_LS: float = [1, 19] # 3.4. maximum number of years in lifestage
    numStem: float = [700, 700] # 3.5. number of stems per m2
    iniCol_frac: float = 0.6 # 3.6. initial colonization fraction (0-1)
    Cd: float = [1.1, 1.15] # 3.7. drag coefficient
    desMort_thres: float = [400, 400] # 3.9. dessication mortality threshold
    desMort_slope: float = [0.75, 0.75] # 3.10. dessication mortality slope
    floMort_thres: float = [0.4, 0.4] # 3.11. flooding mortality threshold
    floMort_slope: float = [0.25, 0.25]# 3.12. flooding mortality slope
    vel_thres: float = [0.15, 0.25]   # 3.13. flow velocity threshold
    vel_slope: float = [3, 3]  # 3.14. flow velocity slope
    maxH_winter: float = [0.4, 0.4]  # 3.15  max height during winter time
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


    @staticmethod
    def get_duration(start_date, end_date):
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return end - start

    @property
    def ets_duration(self):
        return 365 / self.ets_per_year

