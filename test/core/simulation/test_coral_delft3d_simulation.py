import pytest
from src.core.simulation.base_simulation import BaseSimulation
from src.core.simulation.coral_delft3d_simulation import (
    CoralDimrSimulation,
    CoralFlowFmSimulation,
    _CoralDelft3DSimulation,
)


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
        test_coral = coral_mode()
        assert issubclass(coral_mode, _CoralDelft3DSimulation)
        assert issubclass(coral_mode, BaseSimulation)
        assert test_coral.mode == expected_mode
