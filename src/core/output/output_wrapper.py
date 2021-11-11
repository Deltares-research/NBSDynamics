from pathlib import Path

import numpy as np
from src.core.output.output_model import HisOutput, MapOutput
from src.core.output.output_protocol import OutputProtocol
from src.core.base_model import BaseModel
from typing import Iterable, List, Optional, Tuple, Union
from pydantic import root_validator
from datetime import datetime


class OutputWrapper(BaseModel):
    """
    Output files based on predefined output content.
    Generate output files of CoralModel simulation. Output files are formatted as NetCDF4-files.
    """

    output_dir: Path  # directory to write the output to
    xy_coordinates: np.ndarray  # (x,y)-coordinates
    first_date: Union[np.datetime64, datetime]  # first date of simulation

    # Dictionary of values that will be needed to initialize the Output models.
    map_values: dict
    his_values: dict

    # Optional attributes.
    # boolean indicating per (x,y) point if his output is desired
    outpoint: Optional[np.ndarray]

    # Output models.
    map_output: Optional[OutputProtocol]
    his_output: Optional[OutputProtocol]

    @root_validator
    @classmethod
    def check_model(cls, values: dict) -> dict:
        """
        Checks all the provided values and does further assignments if needed.

        Args:
            values (dict): Dictionary of values given to initialize an 'Output'.

        Returns:
            dict: Dictionary of validated values.
        """
        xy_coordinates: np.ndarray = values["xy_coordinates"]
        xy_stations = values.get("xy_stations", None)

        wrap_output_dir: Path = values["output_dir"]
        # Define MapOutput
        map_output: OutputProtocol = values["map_output"]
        if map_output is None:
            values["map_output"] = MapOutput(
                output_dir=wrap_output_dir,
                first_year=values["first_date"].year,
                space=len(xy_coordinates),
                **values["map_values"],
            )

        # Define HisOutput
        his_output: OutputProtocol = values["his_output"]
        if his_output is None:
            xy_stations, idx_stations = cls.get_xy_stations(
                xy_coordinates, values["outpoint"]
            )
            values["his_output"] = HisOutput(
                output_dir=wrap_output_dir,
                first_date=values["first_date"],
                xy_stations=xy_stations,
                idx_stations=idx_stations,
                **values["his_values"],
            )

        return values

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

    @property
    def defined(self) -> bool:
        """Output is defined."""
        return (
            False
            if self.map_output.output_dataset is None
            and self.his_output.output_dataset is None
            else True
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
