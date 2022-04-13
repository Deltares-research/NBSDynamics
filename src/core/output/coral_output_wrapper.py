from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from pydantic.class_validators import root_validator

from src.core.base_model import BaseModel
from src.coral.model.coral_model import Coral
from src.core.output.base_output_wrapper import BaseOutputWrapper
from src.core.output.output_model import HisOutput, MapOutput
from src.core.output.output_protocol import OutputProtocol


class CoralOutputWrapper(BaseOutputWrapper):
    """
    Output files based on predefined output content.
    Generate output files of CoralModel simulation. Output files are formatted as NetCDF4-files.
    """

    # Output models.
    map_output: Optional[MapOutput]
    his_output: Optional[HisOutput]

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
