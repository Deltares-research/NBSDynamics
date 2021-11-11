from src.core.coral_model import Coral
from typing import Protocol
from pathlib import Path

from src.core.output.output_model import ModelParameters


class OutputProtocol(Protocol):
    """
    Protocol defining how an Output model should look like.
    """

    @property
    def output_params(self) -> ModelParameters:
        """
        The output parameters needed to interact with the netcdf dataset.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            ModelParameters: Object with netcdf parameters as attrs..
        """
        raise NotImplementedError

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
    def output_filepath(self) -> Path:
        """
        The full path to the output file.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            Path: Output filepath.
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
