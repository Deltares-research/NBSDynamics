from pathlib import Path
import numpy as np
import pytest
from src.core.output.output_wrapper import OutputWrapper

from src.core.simulation.base_simulation import BaseSimulation
from src.core.simulation.coral_delft3d_simulation import (
    CoralDimrSimulation,
    CoralFlowFmSimulation,
    _CoralDelft3DSimulation,
)
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol


class TestCoralDelft3dSimulation:
    @pytest.mark.parametrize(
        "coral_mode, expected_mode",
        [
            pytest.param(CoralDimrSimulation, "DimrModel", id="Coral DIMR"),
            pytest.param(CoralFlowFmSimulation, "FlowFMModel", id="Coral FlowFM"),
        ],
    )
    def test_delft3d_ctor(
        self, coral_mode: _CoralDelft3DSimulation, expected_mode: str
    ):
        test_coral: _CoralDelft3DSimulation = coral_mode()
        assert issubclass(coral_mode, _CoralDelft3DSimulation)
        assert issubclass(coral_mode, BaseSimulation)
        assert test_coral.mode == expected_mode

    class DummySim(_CoralDelft3DSimulation):
        mode = "TestHydro"

    class DummyHydro(HydrodynamicProtocol):
        config_file = Path.cwd() / "aConfigFile"
        definition_file = Path.cwd() / "aDefFile"
        settings = "someSettings"
        water_depth = np.array([0.42, 0.24])
        space = 42
        x_coordinates = np.array([4, 2])
        y_coordinates = np.array([2, 4])
        xy_coordinates = np.array([[4, 2], [2, 4]])
        init_count = 0
        update_count = 0
        finalise_count = 0

        def initiate(self):
            self.init_count += 1

        def update(self, coral, stormcat: int):
            self.update_count += 1

        def finalise(self):
            self.finalise_count += 1

    def test_configure_hydrodynamics(self):
        """
        This test is only meant to verify that only the initiate
        method gets triggered while configuring hydrodynamics
        at the simulation level.
        """
        hydro_model = self.DummyHydro()
        test_sim = self.DummySim(hydrodynamics=hydro_model)
        assert hydro_model.init_count == 0
        assert hydro_model.update_count == 0
        assert hydro_model.finalise_count == 0

        test_sim.configure_hydrodynamics()
        assert hydro_model.init_count == 1
        assert hydro_model.update_count == 0
        assert hydro_model.finalise_count == 0

    def test_configure_output(self):
        """
        This test is only meant to verify that no init, update or finalize
        methods are triggered while configuring the output.
        """
        hydro_model = self.DummyHydro()
        test_sim = self.DummySim(hydrodynamics=hydro_model)
        assert test_sim.output is None
        assert hydro_model.init_count == 0
        assert hydro_model.update_count == 0
        assert hydro_model.finalise_count == 0

        test_sim.configure_output()

        assert hydro_model.init_count == 0
        assert hydro_model.update_count == 0
        assert hydro_model.finalise_count == 0
        assert isinstance(test_sim.output, OutputWrapper)
        assert test_sim.output.map_output is not None
        assert test_sim.output.his_output is not None
