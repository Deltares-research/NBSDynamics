from typing import Optional

from pydantic import BaseModel

from src.core.biota.biota_model import Biota
from src.core.output.base_output_wrapper import BaseOutputWrapper


class BiotaWrapper(BaseModel):
    """
    Structure containing both `Biota` (input) and its `BaseOutputWrapper` (output)
    """

    biota: Optional[Biota]
    output: Optional[BaseOutputWrapper]
