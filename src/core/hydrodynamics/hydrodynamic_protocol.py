from typing import Protocol

import numpy as np

from src.core.coral_model import Coral


class HydrodynamicProtocol(Protocol):
    """
    Protocol describing the mandatory properties and methods to be implemented by any hydromodel.
    The binding between a model and the protocol is made at the factory level ('HydrodynamicsFactory').
    """

    @property
    def settings(self) -> str:
        """
        Print settings of the hydrodynamic model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            str: The settings as string representation.
        """
        raise NotImplementedError

    @property
    def water_depth(self) -> np.ndarray:
        """
        Water depth, retrieved from hydodynamic model; otherwise base on provided definitino.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            np.ndarray: Water depth value as numpy array.
        """
        raise NotImplementedError

    @property
    def space(self) -> int:
        """
        Space-dimension

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            int: Value of the space-dimension.
        """
        raise NotImplementedError

    @property
    def x_coordinates(self) -> np.ndarray:
        """
        The x-coordinates of the model domain.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            np.ndarray: X coordinates as numpy array.
        """
        raise NotImplementedError

    @property
    def y_coordinates(self) -> np.ndarray:
        """
        The y-coordinates of the model domain.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            np.ndarray: Y coordinates as numpy array.
        """
        raise NotImplementedError

    @property
    def xy_coordinates(self) -> np.ndarray:
        """
        The (x,y)-coordinates of the model domain, retrieved from hydrodynamic model; otherwise based on provided definition.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            np.ndarray: X,Y coordinates as numpy array.
        """
        raise NotImplementedError

    def initiate(self):
        """
        Initiates the working model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError

    def update(self, coral: Coral, stormcat: int):
        """
        Updates the model with the given parameters.

        Args:
            coral (Coral): Coral model to be used.
            stormcat (int): Category of storm to apply.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError

    def finalise(self):
        """
        Finalizes the model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError
