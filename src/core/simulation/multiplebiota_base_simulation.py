from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from pydantic import validator

from src.core.base_model import BaseModel
from src.core.biota.biota_model import Biota
from src.core.common.base_constants import BaseConstants
from src.core.common.environment import Environment
from src.core.hydrodynamics.factory import HydrodynamicsFactory
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.simulation.biota_wrapper import BiotaWrapper


class MultipleBiotaBaseSimulation(BaseModel, ABC):
    """
    Implements the `MultipleBiotaSimulationProtocol`.
    Facade class that can be implemented through an Adapter pattern.

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
    biota_wrapper_list: List[BiotaWrapper]

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
    ) -> Biota:
        pass

    def run(self, duration: Optional[int] = None):
        pass

    def finalise(self):
        """Finalise simulation."""
        pass
