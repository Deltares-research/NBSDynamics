import json
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import root_validator

from src.core.common.base_constants import BaseConstants

default_veg_constants_json = Path(__file__).parent / "veg_constants.json"


class VegetationConstants(BaseConstants):
    """Object containing all constants used in marsh_model simulations."""

    species: str

    input_file: Optional[Path] = default_veg_constants_json

    # Processes
    warn_proc: bool = False
    # User - Define time - scales
    t_eco_year: int = 24  # number ecological time - steps per year(meaning couplings)
    sim_duration: float = 30  # number of morphological years of entire simulation
    start_date: str = "2022-01-01"  # Start date of the simulation

    winter_days: float = list()

    # Colonization
    ColMethod: int = 1  # Colonisation method (1 = on bare substrate between max and min water levels, 2 = on bare substrate with mud content)
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

    @root_validator
    @classmethod
    def post_checks(cls, values: dict) -> dict:
        """
        Class method to validate and check all the fields from the VegetationConstants class.

        Args:
            values (dict): Values already formatted during initialization.

        Returns:
            dict: Dictionary of properties to be set for the VegetationConstants object.
        """
        cls.set_constants_from_default_json(values)
        return values

    @staticmethod
    def set_constants_from_default_json(vegetation_constants: dict):
        """
        Loads all the constants from a given species (inside the 'vegetation_constants' dict), from
        a json file which should be saved in the 'vegetation_constants' dict as 'input_file'.

        Args:
            vegetation_constants (dict): Dictionary to fill with the found values.
        """
        species: str = vegetation_constants["species"]
        with open(vegetation_constants["input_file"]) as f:
            json_data_dict: dict = json.load(f)
            vegetation_constants["ColStart"] = json_data_dict[species]["ColStart"]
            vegetation_constants["ColEnd"] = json_data_dict[species]["ColEnd"]
            vegetation_constants["random"] = json_data_dict[species]["random"]
            vegetation_constants["mud_col"] = json_data_dict[species][
                "mud_colonization"
            ]
            vegetation_constants["fl_dr"] = json_data_dict[species]["fl_dr"]
            vegetation_constants["maxAge"] = json_data_dict[species]["Maximum age"]
            vegetation_constants["num_ls"] = json_data_dict[species][
                "Number LifeStages"
            ]
            vegetation_constants["iniRoot"] = json_data_dict[species][
                "initial root length"
            ]
            vegetation_constants["iniShoot"] = json_data_dict[species][
                "initial shoot length"
            ]
            vegetation_constants["iniDia"] = json_data_dict[species]["initial diameter"]
            vegetation_constants["growth_start"] = json_data_dict[species][
                "start growth period"
            ]
            vegetation_constants["growth_end"] = json_data_dict[species][
                "end growth period"
            ]
            vegetation_constants["winter_start"] = json_data_dict[species][
                "start winter period"
            ]
            vegetation_constants["maxGrowth_H"] = json_data_dict[species][
                "maximum plant height"
            ]
            vegetation_constants["maxDia"] = json_data_dict[species]["maximum diameter"]
            vegetation_constants["maxRoot"] = json_data_dict[species][
                "maximum root length"
            ]
            vegetation_constants["maxYears_LS"] = json_data_dict[species][
                "maximum years in LifeStage"
            ]
            vegetation_constants["num_stem"] = json_data_dict[species]["numStem"]
            vegetation_constants["iniCol_frac"] = json_data_dict[species]["iniCol_frac"]
            vegetation_constants["Cd"] = json_data_dict[species]["Cd"]
            vegetation_constants["desMort_thres"] = json_data_dict[species][
                "desMort_thres"
            ]
            vegetation_constants["desMort_slope"] = json_data_dict[species][
                "desMort_slope"
            ]
            vegetation_constants["floMort_thres"] = json_data_dict[species][
                "floMort_thres"
            ]
            vegetation_constants["floMort_slope"] = json_data_dict[species][
                "floMort_slope"
            ]
            vegetation_constants["vel_thres"] = json_data_dict[species]["vel_thres"]
            vegetation_constants["vel_slope"] = json_data_dict[species]["vel_slope"]
            vegetation_constants["maxH_winter"] = json_data_dict[species]["maxH_winter"]

    @staticmethod
    def get_duration(start_date, end_date):
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return end - start

    @property
    def ets_duration(self):
        return 365 * 24 * 3600 / self.t_eco_year

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
