from pathlib import Path
from typing import Protocol, runtime_checkable

from src.core.vegetation.veg_model import Vegetation
from src.core.output.veg_output_model import ModelParameters


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

    def initialize(self, veg: Vegetation):
        """
        Initializes an output model with the given coral input.

        Args:
           vegetation (Vegetation): Vegetation input model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError

    def update(self, veg: Vegetation, end_time: int):
        """
        Updates the output model with the given vegetation and year.

        Args:
            veg (Vegetation): Vegetation input model.
            period (int): Current period year.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError
