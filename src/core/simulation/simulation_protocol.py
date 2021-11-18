from typing import Protocol, runtime_checkable

from src.core.common.constants import Constants
from src.core.coral.coral_model import Coral
from src.core.common.environment import Environment
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.output.output_wrapper import OutputWrapper


@runtime_checkable
class SimulationProtocol(Protocol):
    """
    Protocol to define simulations for the `NBSDynamics` project.
    """

    @property
    def mode(self) -> str:
        """
        Name of the mode the simulation should run.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            str: Hydrodynamic mode name.
        """
        raise NotImplementedError

    @property
    def hydrodynamics(self) -> HydrodynamicProtocol:
        """
        Instance of hydrodynamic model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            HydrodynamicProtocol: Instantiated object.
        """
        raise NotImplementedError

    @property
    def environment(self) -> Environment:
        """
        Environment in which the simulation takes place.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            Environment: Instantiated environment.
        """
        raise NotImplementedError

    @property
    def constants(self) -> Constants:
        """
        Constants being used for calculations during simulation.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            Constants: Instance of Constants.
        """

    @property
    def coral(self) -> Coral:
        """
        Instance of a Coral model object.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            Coral: Coral instance.
        """
        raise NotImplementedError

    @property
    def output(self) -> OutputWrapper:
        """
        Wrapper containing different output models.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            OutputWrapper: Instance of OutputWrapper.
        """
        raise NotImplementedError

    def initiate(self, x_range: tuple, y_range: tuple, value: float) -> Coral:
        """
        Initiates the simulation attributes with the given parameters.

        Args:
            x_range (tuple): Minimum and maximum x-coordinate.
            y_range (tuple): Minimum and maximum y-coordinate.
            value (float): Coral cover.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            Coral: Initiated coral animal.
        """
        raise NotImplementedError

    def run(self, duration: int):
        """
        Run the simulation with the initiated attributes.

        Args:
            duration (int): Simulation duration [yrs].

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError

    def finalise(self):
        """
        Finalizes simulation

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError
