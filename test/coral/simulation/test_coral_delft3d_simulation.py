from pathlib import Path
from test.utils import TestUtils
from typing import Union

import numpy as np
import pytest

from src.coral.model.coral_constants import CoralConstants
from src.coral.output.coral_output_wrapper import CoralOutputWrapper
from src.coral.simulation.coral_delft3d_simulation import (
    CoralDimrSimulation,
    CoralFlowFmSimulation,
    _CoralDelft3DSimulation,
)
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.simulation.base_simulation import BaseSimulation


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
        assert isinstance(test_sim.output, CoralOutputWrapper)
        assert test_sim.output.map_output is not None
        assert test_sim.output.his_output is not None

    constants_file_case: Path = (
        TestUtils.get_local_test_data_dir("transect_case") / "input" / "coral_input.txt"
    )

    @pytest.mark.parametrize(
        "valid_value",
        [
            pytest.param(CoralConstants(), id="As object"),
            pytest.param(constants_file_case, id="As Path"),
            pytest.param(constants_file_case.as_posix(), id="As String"),
        ],
    )
    def test_validate_coral_constants_with_valid_values(
        self, valid_value: Union[CoralConstants, str, Path]
    ):
        return_value = _CoralDelft3DSimulation.validate_constants(valid_value)
        assert isinstance(return_value, CoralConstants)

    def test_validate_coralconstants_with_not_valid_value(self):
        with pytest.raises(NotImplementedError) as e_err:
            _CoralDelft3DSimulation.validate_constants(42)
        assert str(e_err.value) == f"Validator not available for {type(42)}"
