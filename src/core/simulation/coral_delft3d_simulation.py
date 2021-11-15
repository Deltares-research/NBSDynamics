from abc import ABC

from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.simulation.base_simulation import _Simulation


class _CoralDelft3DSimulation(_Simulation, ABC):
    """
    Coral DDelft3D Simulation. Contains the specific logic and parameters required for the case.
    """

    @classmethod
    def set_simulation_hydrodynamics(
        cls, hydromodel: HydrodynamicProtocol, dict_values: dict
    ):
        """
        Sets the specific hydrodynamic attributes for a `CoralDelft3DSimulation`.

        Args:
            hydromodel (HydrodynamicProtocol): Hydromodel to configure.
            dict_values (dict): Dictionary of values available for assignment.
        """
        hydromodel.working_dir = dict_values.get("working_dir")
        # The following assignments can't be done if None
        if (def_file := dict_values.get("definition_file", None)) is not None:
            hydromodel.definition_file = def_file
        if (con_file := dict_values.get("config_file", None)) is not None:
            hydromodel.config_file = con_file
        if (d3d_home := dict_values.get("d3d_home", None)) is not None:
            hydromodel.d3d_home = d3d_home
        if (upd_intervals := dict_values.get("update_intervals", None)) is not None:
            hydromodel.set_update_intervals(upd_intervals)

    def configure_hydrodynamics(self):
        """
        Configures the hydrodynamics model for a `CoralDelft3DSimulation`.
        """
        self.hydrodynamics.initiate()

    def configure_output(self):
        return


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
