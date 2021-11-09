from pathlib import Path
from typing import Callable

import pytest

from src.core.environment import Constants, Environment
from src.core.hydrodynamics.delft3d import Delft3D
from src.core.hydrodynamics.transect import Transect
from src.core.simulation import Simulation

simulation_cases = [pytest.param("Delft3D"), pytest.param("Transect")]


class TestSimulation:
    @pytest.mark.parametrize(
        "mode_case, expected_hydro",
        [
            pytest.param("Delft3D", Delft3D),
            pytest.param("Transect", Transect),
        ],
    )
    def test_init_simulation_with_supported_modes(
        self, mode_case: str, expected_hydro: Callable
    ):
        test_sim = Simulation(mode=mode_case)
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
        with pytest.raises(ValueError) as e_info:
            Simulation(mode=mode_case)
        assert (
            e_info.value.errors()[0]["msg"]
            == f"{mode_case} not in ['Reef0D', 'Reef1D', 'Delft3D', 'Transect']."
        )

    @pytest.mark.parametrize(
        "unknown_type",
        [
            pytest.param("MAP"),
            pytest.param("HIS"),
            pytest.param("unknown"),
            pytest.param(""),
            pytest.param(None),
        ],
    )
    @pytest.mark.parametrize("mode_case", simulation_cases)
    def test_define_output_with_unknown_output_type_raises(
        self, mode_case: str, unknown_type: str
    ):
        with pytest.raises(ValueError) as e_info:
            test_sim = Simulation(mode=mode_case)
            test_sim.define_output(unknown_type)
        assert str(e_info.value) == f"{unknown_type} not in ('map', 'his')."

    @pytest.mark.parametrize("mode_case", simulation_cases)
    def test_input_check_wihtout_light(self, mode_case: str):
        with pytest.raises(ValueError) as e_info:
            test_sim = Simulation(mode=mode_case)
            assert test_sim.environment.light is None
            test_sim.validate_environment()
        assert (
            str(e_info.value)
            == "CoralModel simulation cannot run without data on light conditions."
        )

    @pytest.mark.parametrize("mode_case", simulation_cases)
    def test_input_check_wihtout_temperature(self, mode_case: str):
        with pytest.raises(ValueError) as e_info:
            test_sim = Simulation(mode=mode_case)
            test_sim.environment._light = 4.2
            assert test_sim.environment.temperature is None
            test_sim.validate_environment()
        assert (
            str(e_info.value)
            == f"CoralModel simulation cannot run without data on temperature conditions."
        )
