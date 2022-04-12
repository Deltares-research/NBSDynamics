from pathlib import Path
from typing import Protocol, runtime_checkable

from src.core.biota.biota_model import Biota
from src.core.output.output_model import ModelParameters


@runtime_checkable
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

    def initialize(self, biota: Biota):
        """
        Initializes an output model with the given biota input.

        Args:
            biota (Biota): Biota input model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError

    def update(self, biota: Biota, year: int):
        """
        Updates the output model with the given biota and year.

        Args:
            biota (Biota): Coral input model.
            year (int): Current calculation year.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError
