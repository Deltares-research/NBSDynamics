"""
coral_model - loop

@author: Gijs G. Hendrickx

"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import validator
from tqdm import tqdm

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
from src.core.coral_model import Coral
from src.core.environment import Environment
from src.core.hydrodynamics.factory import HydrodynamicsFactory
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.output.output_wrapper import OutputWrapper
from src.core.utils import time_series_year


class _Simulation(BaseModel, ABC):
    """CoralModel simulation."""

    mode: str

    # Directories related to working dir
    working_dir: Optional[Path] = Path.cwd()
    figures_dir: Path = working_dir / "figures"
    output_dir: Path = working_dir / "output"
    input_dir: Path = working_dir / "input"

    # Other fields.
    hydrodynamics: Optional[HydrodynamicProtocol]
    environment: Environment = Environment()
    constants: Constants = Constants()
    output: Optional[OutputWrapper]
    coral: Optional[Coral]

    @validator("constants", pre=True)
    @classmethod
    def validate_constants(cls, field_value: Union[str, Path, Constants]) -> Constants:
        """
        Validates the user-input constants value and transforms in case it's a filepath (str, Path).

        Args:
            field_value (Union[str, Path, Constants]): Value given by the user representing Constants.

        Raises:
            NotImplementedError: When the input value does not have any converter.

        Returns:
            Constants: Validated constants value.
        """
        if isinstance(field_value, Constants):
            return field_value
        if isinstance(field_value, str):
            field_value = Path(field_value)
        if isinstance(field_value, Path):
            return Constants.from_input_file(field_value)
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
            hydrodynamics = HydrodynamicsFactory.get_hydrodynamic_model_type(
                field_values.get("mode", values["mode"])
            )()
            # TODO The following logic should actually be sent as arguments into the constructor
            # TODO This will imply creating a base class (like the BaseOutput).
            # Get a merged dictionary.
            sim_dict = dict(list(field_values.items()) + list(values.items()))
            # Emphasize working dir from explicit definition takes preference over simulation one.
            sim_dict["working_dir"] = field_values.get(
                "working_dir", values["working_dir"]
            )
            cls.set_simulation_hydrodynamics(hydrodynamics, sim_dict)

            return hydrodynamics

        return field_values

    @classmethod
    @abstractmethod
    def set_simulation_hydrodynamics(
        cls, hydromodel: HydrodynamicProtocol, dict_values: dict
    ):
        """
        Abstract method that gets triggered during `validate_hydrodynamics_present` so that each `_Simulation` can define extra attributes.

        Args:
            hydromodel (HydrodynamicProtocol): Hydrodynamic model to set up.
            dict_values (dict): Values given by the user to configure the model.

        Raises:
            NotImplementedError: When abstract method not defined in concrete class.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_hydrodynamics(self):
        """
        Configures the parameters for the `HydrodynamicsProtocol`.

        Raises:
            NotImplementedError: [description]
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
        self.coral.RESHAPE.space = self.hydrodynamics.space

        if self.output.defined:
            self.output.initialize(self.coral)
        else:
            msg = "WARNING: No output defined, so none exported."
            print(msg)

        xy = self.hydrodynamics.xy_coordinates

        if value is None:
            value = 1

        cover = value * np.ones(self.coral.RESHAPE.space)

        if x_range is not None:
            x_min = x_range[0] if x_range[0] is not None else min(xy[:][0])
            x_max = x_range[1] if x_range[1] is not None else max(xy[:][0])
            cover[np.logical_or(xy[:][0] <= x_min, xy[:][0] >= x_max)] = 0

        if y_range is not None:
            y_min = y_range[0] if y_range[0] is not None else min(xy[:][1])
            y_max = y_range[1] if y_range[1] is not None else max(xy[:][1])
            cover[np.logical_or(xy[:][1] <= y_min, xy[:][1] >= y_max)] = 0

        self.coral.initiate_spatial_morphology(cover)

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
                self.coral.RESHAPE.time = len(
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
                    constants=self.constants,
                    light_in=time_series_year(self.environment.light, years[i]),
                    lac=time_series_year(self.environment.light_attenuation, years[i]),
                    depth=self.hydrodynamics.water_depth,
                    datareshape=self.coral.RESHAPE,
                )
                lme.rep_light(self.coral)
                # flow micro-environment
                fme = Flow(
                    constants=self.constants,
                    u_current=current_vel,
                    u_wave=wave_vel,
                    h=self.hydrodynamics.water_depth,
                    peak_period=wave_per,
                    datareshape=self.coral.RESHAPE,
                )
                fme.velocities(self.coral, in_canopy=self.constants.fme)
                fme.thermal_boundary_layer(self.coral)
                # thermal micro-environment
                tme = Temperature(
                    constants=self.constants,
                    temperature=time_series_year(
                        self.environment.temp_kelvin, years[i]
                    ),
                    datareshape=self.coral.RESHAPE,
                )
                tme.coral_temperature(self.coral)

                # # physiology
                progress.set_postfix(inner_loop="coral physiology")
                # photosynthetic dependencies
                phd = Photosynthesis(
                    constants=self.constants,
                    light_in=time_series_year(self.environment.light, years[i]),
                    first_year=True if i == 0 else False,
                    datareshape=self.coral.RESHAPE,
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
                    datareshape=self.coral.RESHAPE,
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
                            datareshape=self.coral.RESHAPE,
                        )
                        sfe.velocities(self.coral, in_canopy=self.constants.fme)
                        # storm dislodgement criterion
                        sdc = Dislodgement(self.constants)
                        sdc.update(self.coral)

                # # recruitment
                progress.set_postfix(inner_loop="coral recruitment")
                # recruitment
                rec = Recruitment(self.constants)
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


class CoralTransectSimulation(_Simulation):
    """
    Coral Transect Simulation. Contains the specific logic and parameters required for the case.
    """

    mode = "Transect"

    @classmethod
    def set_simulation_hydrodynamics(
        cls, hydromodel: HydrodynamicProtocol, dict_values: dict
    ):
        """
        Sets the specific hydrodynamic attributes for a `CoralTransectSimulation`.

        Args:
            hydromodel (HydrodynamicProtocol): Hydromodel to configure.
            dict_values (dict): Dictionary of values available for assignment.
        """
        hydromodel.working_dir = dict_values.get("working_dir")
        # The following assignments can't be done if None
        if (def_file := dict_values.get("definition_file", None)) is not None:
            hydromodel.definition_file = def_file
        if (con_file := dict_values.get("config_file", None)) is not None:
            hydromodel.config_file = con_file

    def configure_hydrodynamics(self):
        """
        Initializes the `HydrodynamicsProtocol` model.
        """
        self.hydrodynamics.initiate()

    def configure_output(self):
        """
        Sets the Coral `Transect` specific values to the `OutputWrapper`.
        Should be run after `configure_hydrodynamics`.
        """
        # Initialize the OutputWrapper
        first_date = self.environment.get_dates()[0]
        xy_coordinates = self.hydrodynamics.xy_coordinates
        outpoint = self.hydrodynamics.outpoint

        def get_output_wrapper_dict() -> dict:
            return dict(
                first_date=first_date,
                xy_coordinates=xy_coordinates,
                outpoint=outpoint,
                output_dir=self.working_dir / "output",
            )

        def get_map_output_dict(output_dict: dict) -> dict:
            return dict(
                output_dir=output_dict["output_dir"],
                first_year=output_dict["first_date"].year,
                xy_coordinates=output_dict["xy_coordinates"],
            )

        def get_his_output_dict(output_dict: dict) -> dict:
            xy_stations, idx_stations = OutputWrapper.get_xy_stations(
                output_dict["xy_coordinates"], output_dict["outpoint"]
            )
            return dict(
                output_dir=output_dict["output_dir"],
                first_date=output_dict["first_date"],
                xy_stations=xy_stations,
                idx_stations=idx_stations,
            )

        extended_output = get_output_wrapper_dict()
        map_dict = get_map_output_dict(extended_output)
        his_dict = get_his_output_dict(extended_output)
        if self.output is None:
            extended_output["map_output"] = map_dict
            extended_output["his_output"] = his_dict
            self.output = OutputWrapper(**extended_output)
            return

        def update_output(out_model, new_values: dict):
            if out_model is None:
                return None
            output_dict: dict = out_model.dict()
            for k, v in new_values.items():
                if output_dict.get(k, None) is None:
                    setattr(out_model, k, v)

        update_output(self.output, extended_output)
        update_output(self.output.map_output, map_dict)
        update_output(self.output.his_output, his_dict)


class CoralDelft3DSimulation(_Simulation):
    """
    Coral DDelft3D Simulation. Contains the specific logic and parameters required for the case.
    """

    mode = "Delft3D"

    @classmethod
    def set_simulation_hydrodynamics(
        cls, hydromodel: HydrodynamicProtocol, dict_values: dict
    ):
        """
        Sets the specific hydrodynamic attributes for a `CoralDelft3DSimulation`.

        Args:
            hydromodel (HydrodynamicProtocol): Hydromodel to configure.
            dict_values (dict): Dictionary of values available for assignment.
        """
        hydromodel.working_dir = dict_values.get("working_dir")
        # The following assignments can't be done if None
        if (def_file := dict_values.get("definition_file", None)) is not None:
            hydromodel.definition_file = def_file
        if (con_file := dict_values.get("config_file", None)) is not None:
            hydromodel.config_file = con_file
        if (upd_intervals := dict_values.get("update_intervals", None)) is not None:
            hydromodel.set_update_intervals(upd_intervals)

    def configure_hydrodynamics(self):
        """
        Configures the hydrodynamics model for a `CoralDelft3DSimulation`.
        """
        self.hydrodynamics.initiate()

    def configure_output(self):
        return


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
