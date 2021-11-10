"""
coral_model - loop

@author: Gijs G. Hendrickx

"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import root_validator, validator
from tqdm import tqdm

from src.core import coral_model
from src.core.base_model import BaseModel
from src.core.bio_process.calcification import Calcification
from src.core.bio_process.dislodgment import Dislodgement
from src.core.bio_process.flow import Flow
from src.core.bio_process.light import Light
from src.core.bio_process.morphology import Morphology
from src.core.bio_process.photosynthesis import Photosynthesis
from src.core.bio_process.population_states import PopulationStates
from src.core.bio_process.recruitment import Recruitment
from src.core.bio_process.temperature import Temperature
from src.core.constants import Constants
from src.core.environment import Environment
from src.core.hydrodynamics.factory import HydrodynamicsFactory
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.output_model import Output
from src.core.utils import time_series_year


class Simulation(BaseModel):
    """CoralModel simulation."""

    mode: str

    # Directories related to working dir
    working_dir: Optional[Path] = Path.cwd()
    figures_dir: Path = working_dir / "figures"
    output_dir: Path = working_dir / "output"
    input_dir: Path = working_dir / "input"

    # Other attributes.
    environment: Optional[Environment] = Environment()
    constants: Optional[Constants] = Constants()
    output: Optional[Output] = None
    hydrodynamics: Optional[HydrodynamicProtocol] = None

    @validator("working_dir", always=True)
    @classmethod
    def validate_working_dir(cls, field_value: Optional[Path]) -> Path:
        if field_value is None:
            field_value = Path.cwd()
        if not isinstance(field_value, Path):
            field_value = Path(field_value)

        return field_value

    @root_validator
    @classmethod
    def validate_simulation_attrs(cls, values: dict) -> dict:
        values["hydrodynamics"] = HydrodynamicsFactory.get_hydrodynamic_model(
            values.get("mode", None)
        )
        return values

    def validate_simulation_directories(self):
        """
        Generates the required directories if they do not exist already.
        """
        loop_dirs: List[Path] = [
            "working_dir",
            "output_dir",
            "input_dir",
            "figures_dir",
        ]
        for loop_dir in loop_dirs:
            value_dir: Path = getattr(self, loop_dir)
            if not value_dir.is_dir():
                value_dir.mkdir(parents=True)

    def validate_environment(self):
        """Check input; if all required data is provided."""
        if self.environment.light is None:
            msg = f"CoralModel simulation cannot run without data on light conditions."
            raise ValueError(msg)

        if self.environment.temperature is None:
            msg = f"CoralModel simulation cannot run without data on temperature conditions."
            raise ValueError(msg)

        if self.environment.light_attenuation is None:
            self.environment.set_parameter_values(
                "light_attenuation", self.constants.Kd0
            )
            print(
                f"Light attenuation coefficient set to default: Kd = {self.constants.Kd0} [m-1]"
            )

        if self.environment.aragonite is None:
            self.environment.set_parameter_values("aragonite", self.constants.omegaA0)
            print(
                f"Aragonite saturation state set to default: omega_a0 = {self.constants.omegaA0} [-]"
            )

        # TODO: add other dependencies based on process switches in self.constants if required

    def initiate(
        self,
        coral: coral_model.Coral,
        x_range: Optional[tuple] = None,
        y_range: Optional[tuple] = None,
        value: Optional[float] = None,
    ) -> coral_model.Coral:
        """Initiate the coral distribution. The default coral distribution is a full coral cover over the whole domain.
        More complex initial conditions of the coral cover cannot be realised with this method. See the documentation on
        workarounds to achieve this anyway.

        :param coral: coral animal
        :param x_range: minimum and maximum x-coordinate, defaults to None
        :param y_range: minimum and maximum y-coordinate, defaults to None
        :param value: coral cover, defaults to None

        :type coral: Coral
        :type x_range: tuple, optional
        :type y_range: tuple, optional
        :type value: float, optional

        :return: coral animal initiated
        :rtype: Coral
        """
        # Load constants and validate environment.
        self.validate_simulation_directories()
        self.validate_environment()
        coral.RESHAPE.space = self.hydrodynamics.space

        if self.output.defined:
            self.output.initiate_his()
            self.output.initiate_map(coral)
        else:
            msg = f"WARNING: No output defined, so none exported."
            print(msg)

        xy = self.hydrodynamics.xy_coordinates

        if value is None:
            value = 1

        cover = value * np.ones(coral.RESHAPE.space)

        if x_range is not None:
            x_min = x_range[0] if x_range[0] is not None else min(xy[:][0])
            x_max = x_range[1] if x_range[1] is not None else max(xy[:][0])
            cover[np.logical_or(xy[:][0] <= x_min, xy[:][0] >= x_max)] = 0

        if y_range is not None:
            y_min = y_range[0] if y_range[0] is not None else min(xy[:][1])
            y_max = y_range[1] if y_range[1] is not None else max(xy[:][1])
            cover[np.logical_or(xy[:][1] <= y_min, xy[:][1] >= y_max)] = 0

        coral.initiate_spatial_morphology(cover)

        self.output.initiate_his()
        self.output.initiate_map(coral)

        return coral

    def run(self, coral: coral_model.Coral, duration: Optional[int] = None):
        """Run simulation.

        :param coral: coral animal
        :param duration: simulation duration [yrs], defaults to None

        :type coral: Coral
        :type duration: int, optional
        """
        # auto-set duration based on environmental time-series
        environment_dates: pd.core.series.Series = self.environment.get_dates()
        if duration is None:
            duration = int(
                environment_dates.iloc[-1].year - environment_dates.iloc[0].year
            )
        years = range(
            int(environment_dates.iloc[0].year),
            int(environment_dates.iloc[0].year + duration),
        )

        with tqdm(range((int(duration)))) as progress:
            for i in progress:
                # set dimensions (i.e. update time-dimension)
                coral.RESHAPE.time = len(
                    environment_dates.dt.year[environment_dates.dt.year == years[i]]
                )

                # if-statement that encompasses all for which the hydrodynamic should be used
                progress.set_postfix(inner_loop=f"update {self.hydrodynamics}")
                current_vel, wave_vel, wave_per = self.hydrodynamics.update(
                    coral, stormcat=0
                )

                # # environment
                progress.set_postfix(inner_loop="coral environment")
                # light micro-environment
                lme = Light(
                    constants=self.constants,
                    light_in=time_series_year(self.environment.light, years[i]),
                    lac=time_series_year(self.environment.light_attenuation, years[i]),
                    depth=self.hydrodynamics.water_depth,
                    datareshape=coral.RESHAPE,
                )
                lme.rep_light(coral)
                # flow micro-environment
                fme = Flow(
                    constants=self.constants,
                    u_current=current_vel,
                    u_wave=wave_vel,
                    h=self.hydrodynamics.water_depth,
                    peak_period=wave_per,
                    datareshape=coral.RESHAPE,
                )
                fme.velocities(coral, in_canopy=self.constants.fme)
                fme.thermal_boundary_layer(coral)
                # thermal micro-environment
                tme = Temperature(
                    constants=self.constants,
                    temperature=time_series_year(
                        self.environment.temp_kelvin, years[i]
                    ),
                    datareshape=coral.RESHAPE,
                )
                tme.coral_temperature(coral)

                # # physiology
                progress.set_postfix(inner_loop="coral physiology")
                # photosynthetic dependencies
                phd = Photosynthesis(
                    constants=self.constants,
                    light_in=time_series_year(self.environment.light, years[i]),
                    first_year=True if i == 0 else False,
                    datareshape=coral.RESHAPE,
                )
                phd.photo_rate(coral, self.environment, years[i])
                # population states
                ps = PopulationStates(constants=self.constants)
                ps.pop_states_t(coral)
                # calcification
                cr = Calcification(constants=self.constants)
                cr.calcification_rate(
                    coral, time_series_year(self.environment.aragonite, years[i])
                )
                # # morphology
                progress.set_postfix(inner_loop="coral morphology")
                # morphological development
                mor = Morphology(
                    constants=self.constants,
                    calc_sum=coral.calc.sum(axis=1),
                    light_in=time_series_year(self.environment.light, years[i]),
                    datareshape=coral.RESHAPE,
                )
                mor.update(coral)

                # # storm damage
                if self.environment.storm_category is not None:
                    tt = self.environment.storm_category
                    yr = years[i]
                    stormcat = int(tt["stormcat"].values[tt.index == yr])
                    if stormcat > 0:
                        progress.set_postfix(inner_loop="storm damage")
                        # update hydrodynamic model
                        current_vel, wave_vel, wave_per = self.hydrodynamics.update(
                            coral, stormcat
                        )
                        # storm flow environment
                        sfe = Flow(
                            constants=self.constants,
                            u_current=current_vel,
                            u_wave=wave_vel,
                            h=self.hydrodynamics.water_depth,
                            peak_period=wave_per,
                            datareshape=coral.RESHAPE,
                        )
                        sfe.velocities(coral, in_canopy=self.constants.fme)
                        # storm dislodgement criterion
                        sdc = Dislodgement(self.constants)
                        sdc.update(coral)

                # # recruitment
                progress.set_postfix(inner_loop="coral recruitment")
                # recruitment
                rec = Recruitment(self.constants)
                rec.update(coral)

                # # export results
                progress.set_postfix(inner_loop="export results")
                # map-file
                self.output.update_map(coral, years[i])
                # his-file
                self.output.update_his(
                    coral,
                    environment_dates[environment_dates.dt.year == years[i]],
                )

    def finalise(self):
        """Finalise simulation."""
        self.hydrodynamics.finalise()


class CoralTransectSimulation(Simulation):
    """
    Coral Transect Simulation. Contains the specific logic and parameters
    required for the case.
    """

    mode: str = "Transect"

    # Constant variables
    constants_filename: Path

    # Environment variables
    light: Path
    temperature: Path
    storm: Path
    start_date: str
    end_date: str

    # Hydrodynamics variables
    definition_file: Path
    config_file: Path

    # Other variables.
    output_map_values: Optional[dict] = dict()
    output_his_values: Optional[dict] = dict()

    @root_validator
    @classmethod
    def initialize_coral_transect_simulation_attrs(cls, values: dict) -> dict:
        # Initialize constants.
        constants = Constants.from_input_file(values["constants_filename"])

        # Initialize environment.
        environment = Environment(
            **dict(
                light=values["light"],
                temperature=values["temperature"],
                storm=values["storm"],
                dates=(values["start_date"], values["end_date"]),
            )
        )

        # Initialize hydrodynamics model.
        hydromodel: HydrodynamicProtocol = values["hydrodynamics"]
        hydromodel.working_dir = values["working_dir"]
        hydromodel.definition_file = values["definition_file"]
        hydromodel.config_file = values["config_file"]
        hydromodel.initiate()

        # Initialize output.
        output_model: Output = values["output"]
        if output_model is None:
            output_model = Output(
                values["output_dir"],
                hydromodel.xy_coordinates,
                hydromodel.outpoint,
                environment.get_dates()[0],
            )
        output_model.define_output("map", **values["output_map_values"])
        output_model.define_output("his", **values["output_his_values"])

        # Set formatted values and return.
        values["output"] = output_model
        values["hydrodynamics"] = hydromodel
        values["environment"] = environment
        values["constants"] = constants

        return values


class VersionSimulation(BaseModel):

    environment: Environment
    constants: Constants
    hydrodynamics: HydrodynamicProtocol
    coral: coral_model.Coral
    output: Output

    def initiate(self, duration: Optional[int] = None):
        raise NotImplementedError

    def run(self, duration: Optional[int] = None):
        raise NotImplementedError

    def finalise(self):
        raise NotImplementedError


# TODO: Define folder structure
#  > working directory
#  > figures directory
#  > input directory
#  > output directory
#  > etc.

# TODO: Model initiation IV: OutputFiles
#  > specify output files (i.e. define file names and directories)
#  > specify model data to be included in output files

# TODO: Model initiation V: initial conditions
#  > specify initial morphology
#  > specify initial coral cover
#  > specify carrying capacity

# TODO: Model simulation I: specify SpaceTime

# TODO: Model simulation II: hydrodynamic module
#  > update hydrodynamics
#  > extract variables

# TODO: Model simulation III: coral environment
#  > light micro-environment
#  > flow micro-environment
#  > temperature micro-environment

# TODO: Model simulation IV: coral physiology
#  > photosynthesis
#  > population states
#  > calcification

# TODO: Model simulation V: coral morphology
#  > morphological development

# TODO: Model simulation VI: storm damage
#  > set variables to hydrodynamic module
#  > update hydrodynamics and extract variables
#  > update coral storm survival

# TODO: Model simulation VII: coral recruitment
#  > update recruitment's contribution

# TODO: Model simulation VIII: return morphology
#  > set variables to hydrodynamic module

# TODO: Model simulation IX: export output
#  > write map-file
#  > write his-file

# TODO: Model finalisation
