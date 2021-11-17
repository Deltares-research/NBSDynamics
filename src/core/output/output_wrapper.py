from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from pydantic.class_validators import root_validator

from src.core.base_model import BaseModel
from src.core.coral.coral_model import Coral
from src.core.output.output_model import HisOutput, MapOutput
from src.core.output.output_protocol import OutputProtocol


class OutputWrapper(BaseModel):
    """
    Output files based on predefined output content.
    Generate output files of CoralModel simulation. Output files are formatted as NetCDF4-files.
    """

    output_dir: Path = Path.cwd() / "output"  # directory to write the output to
    xy_coordinates: Optional[np.ndarray]  # (x,y)-coordinates
    first_date: Optional[Union[np.datetime64, datetime]]  # first date of simulation
    outpoint: Optional[
        np.ndarray
    ]  # boolean indicating per (x,y) point if his output is desired

    # Output models.
    map_output: Optional[MapOutput]
    his_output: Optional[HisOutput]

    def __str__(self):
        """String-representation of Output."""
        return (
            f"Output exported:\n\t{self.map_output}\n\t{self.his_output}"
            if self.defined
            else "Output undefined."
        )

    def __repr__(self):
        """Representation of Output."""
        return f"Output(xy_coordinates={self.xy_coordinates}, first_date={self.first_date})"

    @root_validator(pre=True)
    @classmethod
    def check_output_dir(cls, values: dict) -> dict:
        """
        Checks an `output_dir` attribute is given for the `output_model` instances
        `map_output` and `his_output`.

        Args:
            values (dict): Dictionary of attribute values given by the user.

        Returns:
            dict: Dictionary of attribute values with a valid `output_dir` value.
        """

        def check_output_model_dir(out_model: dict) -> Optional[dict]:
            if out_model is None:
                return None
            out_dir_val = out_model.get("output_dir", None)
            if out_dir_val is None:
                out_model["output_dir"] = values["output_dir"]
            return out_model

        values["map_output"] = check_output_model_dir(values["map_output"])
        values["his_output"] = check_output_model_dir(values["his_output"])
        return values

    @property
    def defined(self) -> bool:
        """Output is defined."""

        def output_model_defined(out_model: OutputProtocol) -> bool:
            if out_model is None:
                return False
            return (
                out_model.output_params is not None
                and out_model.output_filepath.exists()
            )

        return output_model_defined(self.map_output) or output_model_defined(
            self.his_output
        )

    @staticmethod
    def get_xy_stations(
        xy_coordinates: np.ndarray, outpoint: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine space indices based on the (x,y)-coordinates of the stations.

        Args:
            xy_coordinates (np.ndarray): Input xy-coordinates system.
            outpoint (np.ndarray): Boolean per x-y indicating if his output is desired.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Resulting tuple of xy_stations, idx_stations
        """
        nout_his = len(xy_coordinates[outpoint, 0])

        x_coord = xy_coordinates[:, 0]
        y_coord = xy_coordinates[:, 1]

        x_station = xy_coordinates[outpoint, 0]
        y_station = xy_coordinates[outpoint, 1]

        idx = np.zeros(nout_his)

        for s in range(len(idx)):
            idx[s] = np.argmin(
                (x_coord - x_station[s]) ** 2 + (y_coord - y_station[s]) ** 2
            )

        idx_stations = idx.astype(int)
        return xy_coordinates[idx_stations, :], idx_stations

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
