from pathlib import Path
from src.core.coral_model import Coral
from typing import Protocol
from netCDF4 import Dataset


class OutputProtocol(Protocol):
    """
    Protocol defining how an Output model should look like.
    """

    @property
    def file_name(self) -> Path:
        raise NotImplementedError

    @property
    def output_dataset(self) -> Dataset:
        raise NotImplementedError

    def initialize(self, coral: Coral):
        raise NotImplementedError

    def update(self, coral: Coral, year: int):
        raise NotImplementedError
