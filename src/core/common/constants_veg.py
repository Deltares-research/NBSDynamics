import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from pydantic import root_validator, validator
from src.core.base_model import BaseModel
from datetime import datetime
from datetime import timedelta
from src.core.common import fpath_constants_file


class Constants(BaseModel):
    """Object containing all constants used in marsh_model simulations."""

    # Input file
    input_file: Optional[Path]
    # Processes
    warn_proc: bool = False
    # User - Define time - scales
    t_eco_year: int = 12 # number ecological time - steps per year(meaning couplings)
    ## TODO check with MorFac, what years is this then?
    sim_duration: float = 30  # number of morphological years of entire simulation
    start_date: str = "2022-01-01"  # Start date of the simulation

    winter_days: float = list()

    # Colonization
    ColMethod: int = 1  # Colonisation method (1 = on bare substrate between max and min water levels, 2 = on bare substrate with mud content)
    species: str = "Spartina anglica"
    # Species Specific Constants
    def __init__(self, species):
        super().__init__()
        self.species = species
        self.get_constants(self.species)
    ColStart: str = None
    ColEnd: str = None
    random: int = None
    mud_col: float = None
    fl_dr: float = None
    maxAge: int = None
    num_ls: int = None
    iniRoot: float = None
    iniShoot: float = None
    iniDia: float = None
    growth_start: str = None
    growth_end: str = None
    winter_start: str = None
    maxGrowth_H: float = None
    maxDia: float = None
    maxRoot: float = None
    maxYears_LS: int = None
    num_stem: int = None
    iniCol_frac: float = None
    Cd: float = None
    desMort_thres: float = None
    desMort_slope: float = None
    floMort_thres: float = None
    floMort_slope: float = None
    vel_thres: float = None
    vel_slope: float = None
    maxH_winter: float = None


    def get_constants(self, species):
        # if species_1 is not None and species_2 is None:
        with open(fpath_constants_file) as f:
            constants_dict = json.load(f)
            self.ColStart = constants_dict[species]['ColStart']
            self.ColEnd = constants_dict[species]['ColEnd']
            self.random = constants_dict[species]['random']
            self.mud_col = constants_dict[species]['mud_colonization']
            self.fl_dr = constants_dict[species]['fl_dr']
            self.maxAge = constants_dict[species]['Maximum age']
            self.num_ls = constants_dict[species]['Number LifeStages']
            self.iniRoot = constants_dict[species]['initial root length']
            self.iniShoot = constants_dict[species]['initial shoot length']
            self.iniDia = constants_dict[species]['initial diameter']
            self.growth_start = constants_dict[species]['start growth period']
            self.growth_end = constants_dict[species]['end growth period']
            self.winter_start = constants_dict[species]['start winter period']
            self.maxGrowth_H = constants_dict[species]['maximum plant height']
            self.maxDia = constants_dict[species]['maximum diameter']
            self.maxRoot = constants_dict[species]['maximum root length']
            self.maxYears_LS = constants_dict[species]['maximum years in LifeStage']
            self.num_stem = constants_dict[species]['numStem']
            self.iniCol_frac = constants_dict[species]['iniCol_frac']
            self.Cd = constants_dict[species]['Cd']
            self.desMort_thres = constants_dict[species]['desMort_thres']
            self.desMort_slope = constants_dict[species]['desMort_slope']
            self.floMort_thres = constants_dict[species]['floMort_thres']
            self.floMort_slope = constants_dict[species]['floMort_slope']
            self.vel_thres = constants_dict[species]['vel_thres']
            self.vel_slope = constants_dict[species]['vel_slope']
            self.maxH_winter = constants_dict[species]['maxH_winter']

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
        # cls_constants.correct_values()
        return cls_constants

    @staticmethod
    def get_duration(start_date, end_date):
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return end - start

    @property
    def ets_duration(self):
        return round(365 / self.t_eco_year)

    # @property
    # def growth_days(self):
    #         """
    #         find number of growth days in current ets depending on start and end of growth period
    #         """
    #         ##TODO get rid of loops
    #         current_date = pd.to_datetime(self.start_date)
    #         growth_days = []
    #         for x in range(0, self.t_eco_year):
    #             growth_Day = []
    #             for y in range(0, round(self.ets_duration)):
    #                 if pd.to_datetime(self.growth_start).month <= current_date.month <= pd.to_datetime(
    #                         self.growth_end).month:
    #                     if pd.to_datetime(self.growth_start).month == current_date.month:
    #                         if pd.to_datetime(self.growth_start).day <= current_date.day:
    #                             growth_Day.append(1)
    #                         else:
    #                             growth_Day.append(0)
    #                     elif pd.to_datetime(self.growth_end).month == current_date.month:
    #                         if current_date.day <= pd.to_datetime(self.growth_end).day:
    #                             growth_Day.append(1)
    #                         else:
    #                             growth_Day.append(0)
    #                     else:
    #                         growth_Day.append(1)
    #                 else:
    #                     growth_Day.append(0)
    #                 current_date = current_date + timedelta(days=1)
    #
    #             growth_days.append(sum(growth_Day))
    #         return np.array(growth_days)

    # @property
    # def col_days(self):
    #         """
    #         find ets where colonization happens
    #         """
    #         ##TODO get rid of loops
    #         days_ets = 365 / self.t_eco_year
    #         current_date = pd.to_datetime(self.start_date)
    #         col_days = []
    #         for x in range(0, self.t_eco_year):
    #             col_Day = []
    #             for y in range(0, round(days_ets)):
    #                 if pd.to_datetime(self.ColStart).month <= current_date.month <= pd.to_datetime(
    #                         self.ColEnd).month:
    #                     if pd.to_datetime(self.ColStart).month == current_date.month:
    #                         if pd.to_datetime(self.ColStart).day <= current_date.day:
    #                             col_Day.append(1)
    #                         else:
    #                             col_Day.append(0)
    #                     elif pd.to_datetime(self.ColEnd).month == current_date.month:
    #                         if current_date.day <= pd.to_datetime(self.ColEnd).day:
    #                             col_Day.append(1)
    #                         else:
    #                             col_Day.append(0)
    #                     else:
    #                         col_Day.append(1)
    #                 else:
    #                     col_Day.append(0)
    #                 current_date = current_date + timedelta(days=1)
    #
    #             col_days.append(sum(col_Day))
    #         return np.array(col_days)

    # def get_WinterDays(self, constants):
    #         """
    #         find number of winter days in current ets depending on start and end of growth period
    #         """
    #         ##TODO get rid of loops
    #         current_date = pd.to_datetime(constants.winter_start)
    #         winter_days = []
    #         for x in range(0, constants.t_eco_year):
    #             winter_Day = []
    #             for y in range(0, round(constants.ets_duration)):
    #                 if pd.to_datetime(constants.winter_start).month <= current_date.month <= pd.to_datetime(
    #                         constants.growth_start).month:
    #                     if pd.to_datetime(constants.winter_start).month == current_date.month:
    #                         if pd.to_datetime(constants.winter_start).day <= current_date.day:
    #                             winter_Day.append(1)
    #                         else:
    #                             winter_Day.append(0)
    #                     elif pd.to_datetime(constants.growth_start).month == current_date.month:
    #                         if current_date.day <= pd.to_datetime(constants.growth_start).day:
    #                             winter_Day.append(1)
    #                         else:
    #                             winter_Day.append(0)
    #                     else:
    #                         winter_Day.append(1)
    #                 else:
    #                     winter_Day.append(0)
    #                 current_date = current_date + timedelta(days=1)
    #
    #             winter_days.append(sum(winter_Day))
    #         return np.array(winter_days)
