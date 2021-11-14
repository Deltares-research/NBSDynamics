from pathlib import Path
from typing import Callable

import pytest

from src.core.constants import Constants
from src.core.environment import Environment
from src.core.hydrodynamics.delft3d import DimrModel, FlowFmModel
from src.core.hydrodynamics.transect import Transect
from src.core.simulation.coral_delft3d_simulation import (
    CoralDimrSimulation,
    CoralFlowFmSimulation,
)
from src.core.simulation.coral_transect_simulation import CoralTransectSimulation
from src.core.simulation.base_simulation import _Simulation

simulation_cases = [
    pytest.param(CoralDimrSimulation),
    pytest.param(CoralFlowFmSimulation),
    pytest.param(CoralTransectSimulation),
]


class TestSimulation:
    @pytest.mark.parametrize(
        "mode_case, expected_hydro",
        [
            pytest.param(CoralFlowFmSimulation, FlowFmModel),
            pytest.param(CoralDimrSimulation, DimrModel),
            pytest.param(CoralTransectSimulation, Transect),
        ],
    )
    def test_init_simulation_with_supported_modes(
        self, mode_case: _Simulation, expected_hydro: Callable
    ):
        test_sim = mode_case()
        assert isinstance(test_sim.environment, Environment)
        assert isinstance(test_sim.constants, Constants)
        assert isinstance(test_sim.working_dir, Path)
        assert test_sim.output is None
        assert isinstance(test_sim.hydrodynamics, expected_hydro)
        assert test_sim.figures_dir.stem == "figures"
        assert test_sim.output_dir.stem == "output"

    @pytest.mark.parametrize(
        "mode_case", [pytest.param(""), pytest.param("unsupported")]
    )
    def test_init_simulation_unsupported_modes(self, mode_case: str):
        with pytest.raises(TypeError):
            _Simulation(mode=mode_case)

    @pytest.mark.parametrize("mode_case", simulation_cases)
    def test_input_check_wihtout_light(self, mode_case: _Simulation):
        with pytest.raises(ValueError) as e_info:
            test_sim = mode_case()
            assert test_sim.environment.light is None
            test_sim.validate_environment()
        assert (
            str(e_info.value)
            == "CoralModel simulation cannot run without data on light conditions."
        )

    @pytest.mark.parametrize("mode_case", simulation_cases)
    def test_input_check_wihtout_temperature(self, mode_case: _Simulation):
        with pytest.raises(ValueError) as e_info:
            test_sim = mode_case()
            test_sim.environment.light = 4.2
            assert test_sim.environment.temperature is None
            test_sim.validate_environment()
        assert (
            str(e_info.value)
            == "CoralModel simulation cannot run without data on temperature conditions."
        )
