import pytest

from src.core.loop import Simulation


class TestSimulation:
    @pytest.mark.parametrize(
        "mode_case",
        [
            pytest.param("Delft3D"),
            pytest.param("Transect"),
        ],
    )
    def test_init_simulation_with_supported_modes(self, mode_case: str):
        assert Simulation(mode=mode_case) is not None

    @pytest.mark.parametrize(
        "mode_case", [pytest.param(""), pytest.param(None), pytest.param("unsupported")]
    )
    def test_init_simulation_unsupported_modes(self, mode_case: str):
        with pytest.raises(ValueError) as e_info:
            Simulation(mode_case)
        assert str(e_info.value) == f"{mode_case} not in [Delft3D, Transect]."
