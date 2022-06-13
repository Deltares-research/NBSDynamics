from typing import Optional

from src.biota_models.mangroves.model.mangrove_model import Mangrove
from src.biota_models.mangroves.output.mangrove_output_model import (
    MangroveHisOutput,
    MangroveMapOutput,
)
from src.core.output.base_output_wrapper import BaseOutputWrapper


class MangroveOutputWrapper(BaseOutputWrapper):
    """
    Output files based on predefined output content.
    Generate output files of VegetationModel simulation. Output files are formatted as NetCDF4-files.
    """

    # Output models.
    map_output: Optional[MangroveMapOutput]
    his_output: Optional[MangroveHisOutput]

    def initialize(self, mangrove_model: Mangrove):
        """
        Initializes all available output models (His and Map).

        Args:
            veg_model (Vegetation): Vegetation model to be used in the output.
        """
        # Initialize Output dir path.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize output models.
        self.his_output.initialize(mangrove_model)
        self.map_output.initialize(mangrove_model)
