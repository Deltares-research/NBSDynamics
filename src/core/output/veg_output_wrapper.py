from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from pydantic.class_validators import root_validator

from src.core.base_model import BaseModel
from src.core.vegetation.veg_model import Vegetation
from src.core.output.veg_output_model import HisOutput, MapOutput
from src.core.output.output_protocol import OutputProtocol


class VegOutputWrapper(BaseModel):

