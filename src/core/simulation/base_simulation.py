from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import validator
from tqdm import tqdm

from src.core import RESHAPE
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
from src.core.biota.coral.coral_model import Coral
from src.core.common.base_constants import BaseConstants
from src.core.common.coral_constants import CoralConstants
from src.core.common.environment import Environment
from src.core.common.space_time import time_series_year
from src.core.hydrodynamics.factory import HydrodynamicsFactory
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.output.base_output_wrapper import BaseOutputWrapper


class BaseSimulation(BaseModel, ABC):
    """
    Implements the `SimulationProtocol`.
    Facade class that can be implemented through an Adapter pattern.
    CoralModel simulation.
    """

    mode: str

    # Directories related to working dir
    working_dir: Optional[Path] = Path.cwd()
    figures_dir: Path = working_dir / "figures"
    output_dir: Path = working_dir / "output"
    input_dir: Path = working_dir / "input"

    # Other fields.
    hydrodynamics: Optional[HydrodynamicProtocol]
    environment: Environment = Environment()
    constants: Optional[BaseConstants]
    output: Optional[BaseOutputWrapper]
    coral: Optional[Coral]

    @validator("constants", pre=True)
    @classmethod
    def validate_constants(
        cls, field_value: Union[str, Path, BaseConstants]
    ) -> BaseConstants:
        """
        Validates the user-input constants value and transforms in case it's a filepath (str, Path).

        Args:
            field_value (Union[str, Path, Constants]): Value given by the user representing Constants.

        Raises:
            NotImplementedError: When the input value does not have any converter.

        Returns:
            Constants: Validated constants value.
        """
        if isinstance(field_value, BaseConstants):
            return field_value
        if isinstance(field_value, str):
            field_value = Path(field_value)
        if isinstance(field_value, Path):
            return CoralConstants.from_input_file(field_value)
        raise NotImplementedError(f"Validator not available for {type(field_value)}")

    @validator("coral", pre=True)
    @classmethod
    def validate_coral(cls, field_value: Union[dict, Coral], values: dict) -> Coral:
        """
        Initializes coral in case a dictionary is provided. Ensuring the constants are also
        given to the object.

        Args:
            field_value (Union[dict, Coral]): Value given by the user for the Coral field.
            values (dict): Dictionary of remaining user-given field values.

        Returns:
            Coral: Validated instance of 'Coral'.
        """
        if isinstance(field_value, Coral):
            return field_value
        if isinstance(field_value, dict):
            # Check if constants present in the dictionary:
            if "constants" in field_value.keys():
                # It will be generated automatically.
                # in case parameters are missing an error will also be displayed.
                return Coral(**field_value)
            if "constants" in values.keys():
                field_value["constants"] = values["constants"]
                return Coral(**field_value)
            raise ValueError(
                "Constants should be provided to initialize a Coral Model."
            )
        raise NotImplementedError(f"Validator not available for {type(field_value)}")

    @validator("hydrodynamics", pre=True, always=True)
    @classmethod
    def validate_hydrodynamics_present(
        cls, field_values: Union[dict, HydrodynamicProtocol], values: dict
    ) -> HydrodynamicProtocol:
        """
        Validator to transform the given dictionary into the corresponding hydrodynamic model.

        Args:
            field_values (Union[dict, HydrodynamicProtocol]): Value assigned to `hydrodynamics`.
            values (dict): Dictionary of values given by the user.

        Raises:
            ValueError: When no hydrodynamics model can be built with the given values.

        Returns:
            dict: Validated dictionary of values given by the user.
        """
        if field_values is None:
            field_values = dict()
        if isinstance(field_values, dict):
            return HydrodynamicsFactory.create(
                field_values.get("mode", values["mode"]), **field_values
            )

        return field_values

    @abstractmethod
    def configure_hydrodynamics(self):
        """
        Configures the parameters for the `HydrodynamicsProtocol`.

        Raises:
            NotImplementedError: When abstract method not defined in concrete class.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_output(self):
        """
        Configures the parameters for the `OutputWrapper`.
        """
        raise NotImplementedError

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
            msg = "CoralModel simulation cannot run without data on light conditions."
            raise ValueError(msg)

        if self.environment.temperature is None:
            msg = "CoralModel simulation cannot run without data on temperature conditions."
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
        x_range: Optional[tuple] = None,
        y_range: Optional[tuple] = None,
        value: Optional[float] = None,
    ) -> Coral:
        """Initiate the coral distribution. The default coral distribution is a full coral cover over the whole domain.
        More complex initial conditions of the coral cover cannot be realised with this method. See the documentation on
        workarounds to achieve this anyway.

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
        self.configure_hydrodynamics()
        self.configure_output()
        # Load constants and validate environment.
        self.validate_simulation_directories()
        self.validate_environment()
        RESHAPE().space = self.hydrodynamics.space

        if self.output.defined:
            self.output.initialize(self.coral)
        else:
            print("WARNING: No output defined, so none exported.")

        xy = self.hydrodynamics.xy_coordinates

        if value is None:
            value = 1

        cover = value * np.ones(RESHAPE().space)

        if x_range is not None:
            x_min = x_range[0] if x_range[0] is not None else min(xy[:][0])
            x_max = x_range[1] if x_range[1] is not None else max(xy[:][0])
            cover[np.logical_or(xy[:][0] <= x_min, xy[:][0] >= x_max)] = 0

        if y_range is not None:
            y_min = y_range[0] if y_range[0] is not None else min(xy[:][1])
            y_max = y_range[1] if y_range[1] is not None else max(xy[:][1])
            cover[np.logical_or(xy[:][1] <= y_min, xy[:][1] >= y_max)] = 0

        self.coral.initiate_coral_morphology(cover)

        self.output.initialize(self.coral)

    def run(self, duration: Optional[int] = None):
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
                RESHAPE().time = len(
                    environment_dates.dt.year[environment_dates.dt.year == years[i]]
                )

                # if-statement that encompasses all for which the hydrodynamic should be used
                progress.set_postfix(inner_loop=f"update {self.hydrodynamics}")
                current_vel, wave_vel, wave_per = self.hydrodynamics.update(
                    self.coral, stormcat=0
                )

                # # environment
                progress.set_postfix(inner_loop="coral environment")
                # light micro-environment
                lme = Light(
                    light_in=time_series_year(self.environment.light, years[i]),
                    lac=time_series_year(self.environment.light_attenuation, years[i]),
                    depth=self.hydrodynamics.water_depth,
                )
                lme.rep_light(self.coral)
                # flow micro-environment
                fme = Flow(
                    u_current=current_vel,
                    u_wave=wave_vel,
                    h=self.hydrodynamics.water_depth,
                    peak_period=wave_per,
                    constants=self.constants,
                )
                fme2 = Flow(
                    u_current=current_vel,
                    u_wave=wave_vel,
                    h=self.hydrodynamics.water_depth,
                    peak_period=wave_per,
                    constants=self.constants,
                )
                fme.velocities(self.coral, in_canopy=self.constants.fme)
                fme.thermal_boundary_layer(self.coral)
                # thermal micro-environment
                tme = Temperature(
                    constants=self.constants,
                    temperature=time_series_year(
                        self.environment.temp_kelvin, years[i]
                    ),
                )
                tme.coral_temperature(self.coral)

                # # physiology
                progress.set_postfix(inner_loop="coral physiology")
                # photosynthetic dependencies
                phd = Photosynthesis(
                    constants=self.constants,
                    light_in=time_series_year(self.environment.light, years[i]),
                    first_year=True if i == 0 else False,
                )
                phd.photo_rate(self.coral, self.environment, years[i])
                # population states
                ps = PopulationStates(constants=self.constants)
                ps.pop_states_t(self.coral)
                # calcification
                cr = Calcification(constants=self.constants)
                cr.calcification_rate(
                    self.coral, time_series_year(self.environment.aragonite, years[i])
                )
                # # morphology
                progress.set_postfix(inner_loop="coral morphology")
                # morphological development
                mor = Morphology(
                    constants=self.constants,
                    calc_sum=self.coral.calc.sum(axis=1),
                    light_in=time_series_year(self.environment.light, years[i]),
                )
                mor.update(self.coral)

                # # storm damage
                if self.environment.storm_category is not None:
                    tt = self.environment.storm_category
                    yr = years[i]
                    stormcat = int(tt["stormcat"].values[tt.index == yr])
                    if stormcat > 0:
                        progress.set_postfix(inner_loop="storm damage")
                        # update hydrodynamic model
                        current_vel, wave_vel, wave_per = self.hydrodynamics.update(
                            self.coral, stormcat
                        )
                        # storm flow environment
                        sfe = Flow(
                            constants=self.constants,
                            u_current=current_vel,
                            u_wave=wave_vel,
                            h=self.hydrodynamics.water_depth,
                            peak_period=wave_per,
                        )
                        sfe.velocities(self.coral, in_canopy=self.constants.fme)
                        # storm dislodgement criterion
                        sdc = Dislodgement(constants=self.constants)
                        sdc.update(self.coral)

                # # recruitment
                progress.set_postfix(inner_loop="coral recruitment")
                # recruitment
                rec = Recruitment(constants=self.constants)
                rec.update(self.coral)

                # # export results
                progress.set_postfix(inner_loop="export results")
                # map-file
                self.output.map_output.update(self.coral, years[i])
                # his-file
                self.output.his_output.update(
                    self.coral,
                    environment_dates[environment_dates.dt.year == years[i]],
                )

    def finalise(self):
        """Finalise simulation."""
        self.hydrodynamics.finalise()


class Simulation(BaseSimulation):
    """
    Vanilla definition of the `BaseSimulation` that allows any user
    to create their flat simulation without pre-defined values.
    In other words, everything should be built manually.
    """

    def configure_hydrodynamics(self):
        """
        This flat Simulation type does not configure anything automatically.
        """
        pass

    def configure_output(self):
        """
        This flat Simulation type does not configure anything automatically.
        """
        pass


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
