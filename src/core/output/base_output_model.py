from abc import ABC
from pathlib import Path
from typing import Optional

from src.core.base_model import BaseModel
from src.core.biota.biota_model import Biota


class BaseOutputParameters(BaseModel, ABC):
    def valid_output(self) -> bool:
        """
        Validates whether all the fields from this class have a value.

        Returns:
            bool: When all the values are 'filled'.
        """
        return any(self.dict().values())


class BaseOutput(BaseModel, ABC):
    """
    Base class containing the generic definition of an output model.
    """

    output_dir: Path
    output_filename: str

    # Output model attributes.
    output_params: BaseOutputParameters = BaseOutputParameters()

    def valid_output(self) -> bool:
        """
        Verifies whether this model can generate valid output.

        Returns:
            bool: Output is valid.
        """
        return self.output_params.valid_output()

    @property
    def output_filepath(self) -> Path:
        """
        Gets the full path to the output netcdf file.

        Returns:
            Path: Output .nc file.
        """
        return self.output_dir / self.output_filename

    def initialize(self, biota: Optional[Biota]):
        """
        Method to initialize the Output Model based on a given biota model.
        This method should be implemented in the concrete classes.

        Args:
            biota (Optional[Biota]): Base model for the generated output.
        """
        pass
