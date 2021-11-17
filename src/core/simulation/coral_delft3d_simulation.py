from abc import ABC
from src.core.hydrodynamics.delft3d import Delft3D

from src.core.output.output_wrapper import OutputWrapper
from src.core.simulation.base_simulation import BaseSimulation


class _CoralDelft3DSimulation(BaseSimulation, ABC):
    """
    Implements the `SimulationProtocol`
    Coral DDelft3D Simulation. Contains the specific logic and parameters required for the case.
    """

    def configure_hydrodynamics(self):
        """
        Configures the hydrodynamics model for a `CoralDelft3DSimulation`.
        """
        self.hydrodynamics.initiate()

    def configure_output(self):
        first_date = self.environment.get_dates()[0]
        xy_coordinates = self.hydrodynamics.xy_coordinates
        outpoint = self.hydrodynamics.outpoint

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
            xy_stations, idx_stations = OutputWrapper.get_xy_stations(
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
            self.output = OutputWrapper(**extended_output)
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


class CoralDimrSimulation(_CoralDelft3DSimulation):
    """
    Coral Dimr Simulation representation. Implements the specific
    logic needed to run a Coral Simulation with a DIMR kernel through
    `BMIWrapper`
    """

    mode = "DimrModel"


class CoralFlowFmSimulation(_CoralDelft3DSimulation):
    """
    Coral FlowFM Simulation representation. Implements the specific
    logic needed to run a Coral Simulation with a FlowFM kernel through
    `BMIWrapper`
    """

    mode = "FlowFMModel"
