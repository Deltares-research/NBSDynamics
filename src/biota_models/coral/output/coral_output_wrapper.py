from typing import Optional

from src.biota_models.coral.model.coral_model import Coral
from src.biota_models.coral.output.coral_output_model import (
    CoralHisOutput,
    CoralMapOutput,
)
from src.core.output.base_output_wrapper import BaseOutputWrapper


class CoralOutputWrapper(BaseOutputWrapper):
    """
    Output files based on predefined output content.
    Generate output files of CoralModel simulation. Output files are formatted as NetCDF4-files.
    """

    # Output models.
    map_output: Optional[CoralMapOutput]
    his_output: Optional[CoralHisOutput]

    def initialize(self, coral: Coral):
        """
        Initializes all available output models (His and Map).

        Args:
            coral (Coral): Coral model to be used in the output.
        """
        # Initialize Output dir path.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize output models.
        self.his_output.initialize(coral)
        self.map_output.initialize(coral)
