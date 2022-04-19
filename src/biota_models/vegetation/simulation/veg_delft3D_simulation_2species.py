from abc import ABC

import pandas as pd

from src.biota_models.vegetation.output.veg_output_wrapper import VegOutputWrapper
from src.biota_models.vegetation.simulation.veg_simulation_2species import (
    _VegetationSimulation_2species,
)
from src.core.hydrodynamics.delft3d import Delft3D


class _VegDelft3DSimulation(_VegetationSimulation_2species, ABC):
    """
    Implements the `MultipleBiotaSimulationProtocol`
    Vegetation Delft3D Simulation. Contains the specific logic and parameters required for the case.
    """

    def configure_hydrodynamics(self):
        """
        Configures the hydrodynamics model for a `VegDelft3DSimulation`.
        """
        self.hydrodynamics.initiate()

    def configure_output(self):
        first_date = pd.to_datetime(self.constants.start_date)
        hydromodel: Delft3D = self.hydrodynamics
        xy_coordinates = hydromodel.xy_coordinates
        # TODO: There should be an output definition for this model.
        # TODO: For now just output all the points.
        outpoint = hydromodel.x_coordinates[:] >= 0

        def get_output_wrapper_dict() -> dict:
            return dict(
                first_date=first_date,
                xy_coordinates=xy_coordinates,
                outpoint=outpoint,
                output_dir=self.working_dir / "output",
            )

        def get_map_output_dict(output_dict: dict) -> dict:
            return dict(
                output_dir=output_dict["output_dir"],
                first_year=output_dict["first_date"].year,
                xy_coordinates=output_dict["xy_coordinates"],
            )

        def get_his_output_dict(output_dict: dict) -> dict:
            xy_stations, idx_stations = VegOutputWrapper.get_xy_stations(
                output_dict["xy_coordinates"], output_dict["outpoint"]
            )
            return dict(
                output_dir=output_dict["output_dir"],
                first_date=output_dict["first_date"],
                xy_stations=xy_stations,
                idx_stations=idx_stations,
            )

        extended_output = get_output_wrapper_dict()
        map_dict = get_map_output_dict(extended_output)
        his_dict = get_his_output_dict(extended_output)
        if self.output is None:
            extended_output["map_output"] = map_dict
            extended_output["his_output"] = his_dict
            self.output = VegOutputWrapper(**extended_output)
            return

        if self.output2 is None:
            extended_output["map_output"] = map_dict
            extended_output["his_output"] = his_dict
            self.output2 = VegOutputWrapper(**extended_output)
            return

        def update_output(out_model, new_values: dict):
            if out_model is None:
                return None
            output_dict: dict = out_model.dict()
            for k, v in new_values.items():
                if output_dict.get(k, None) is None:
                    setattr(out_model, k, v)

        update_output(self.output, extended_output)
        update_output(self.output.map_output, map_dict)
        update_output(self.output.his_output, his_dict)

        update_output(self.output2, extended_output)
        update_output(self.output2.map_output, map_dict)
        update_output(self.output2.his_output, his_dict)


class VegDimrSimulation(_VegDelft3DSimulation):
    """
    Vegetation Dimr Simulation representation. Implements the specific
    logic needed to run a Veg Simulation with a DIMR kernel through
    `BMIWrapper`
    """

    mode = "DimrModel"


class VegFlowFmSimulation_2species(_VegDelft3DSimulation):
    """
    Vegetation FlowFM Simulation representation. Implements the specific
    logic needed to run a Veg Simulation with a FlowFM kernel through
    `BMIWrapper`
    """

    mode = "FlowFMModel"
