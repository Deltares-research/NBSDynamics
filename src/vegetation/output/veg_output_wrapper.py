from typing import Optional

from src.core.output.base_output_wrapper import BaseOutputWrapper
from src.vegetation.model.veg_model import Vegetation
from src.vegetation.output.veg_output_model import HisOutput, MapOutput


class VegOutputWrapper(BaseOutputWrapper):
    """
    Output files based on predefined output content.
    Generate output files of VegetationModel simulation. Output files are formatted as NetCDF4-files.
    """

    # Output models.
    map_output: Optional[MapOutput]
    his_output: Optional[HisOutput]

    def initialize(self, veg_model: Vegetation):
        """
        Initializes all available output models (His and Map).

        Args:
            veg_model (Vegetation): Vegetation model to be used in the output.
        """
        # Initialize Output dir path.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize output models.
        self.his_output.initialize(veg_model)
        self.map_output.initialize()
