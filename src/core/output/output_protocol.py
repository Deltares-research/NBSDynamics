from src.core.coral_model import Coral
from typing import Protocol
from netCDF4 import Dataset


class OutputProtocol(Protocol):
    """
    Protocol defining how an Output model should look like.
    """

    @property
    def output_filename(self) -> str:
        """
        The basename with extension the output file will have.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            str: Output filename.
        """
        raise NotImplementedError

    @property
    def output_dataset(self) -> Dataset:
        """
        The stored netCDF Dataset output for the given model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            Dataset: netCDF Dataset being used to store the results.
        """
        raise NotImplementedError

    def initialize(self, coral: Coral):
        """
        Initializes an output model with the given coral input.

        Args:
            coral (Coral): Coral input model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError

    def update(self, coral: Coral, year: int):
        """
        Updates the output model with the given coral and year.

        Args:
            coral (Coral): Coral input model.
            year (int): Current calculation year.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError
