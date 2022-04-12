from pathlib import Path
from test.utils import TestUtils
from typing import Any, Callable, Optional, Union

import pytest

from src.core.biota.coral.coral_model import Coral
from src.core.common.coral_constants import CoralConstants
from src.core.common.environment import Environment
from src.core.hydrodynamics.delft3d import DimrModel, FlowFmModel
from src.core.hydrodynamics.factory import HydrodynamicsFactory
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.hydrodynamics.transect import Transect
from src.core.simulation.base_simulation import BaseSimulation, Simulation
from src.core.simulation.coral_delft3d_simulation import (
    CoralDimrSimulation,
    CoralFlowFmSimulation,
)
from src.core.simulation.coral_transect_simulation import CoralTransectSimulation
from src.core.simulation.simulation_protocol import SimulationProtocol

simulation_cases = [
    pytest.param(CoralDimrSimulation),
    pytest.param(CoralFlowFmSimulation),
    pytest.param(CoralTransectSimulation),
]


class TestBaseSimulation:
    @pytest.mark.parametrize(
        "mode_case, expected_hydro",
        [
            pytest.param(CoralFlowFmSimulation, FlowFmModel),
            pytest.param(CoralDimrSimulation, DimrModel),
            pytest.param(CoralTransectSimulation, Transect),
        ],
    )
    def test_init_simulation_with_supported_modes(
        self, mode_case: BaseSimulation, expected_hydro: Callable
    ):
        test_sim: BaseSimulation = mode_case()
        assert issubclass(mode_case, BaseSimulation)
        assert isinstance(test_sim, SimulationProtocol)
        assert isinstance(test_sim.environment, Environment)
        assert isinstance(test_sim.constants, CoralConstants)
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
            BaseSimulation(mode=mode_case)

    @pytest.mark.parametrize("mode_case", simulation_cases)
    def test_input_check_without_light(self, mode_case: BaseSimulation):
        with pytest.raises(ValueError) as e_info:
            test_sim: BaseSimulation = mode_case()
            assert test_sim.environment.light is None
            test_sim.validate_environment()
        assert (
            str(e_info.value)
            == "CoralModel simulation cannot run without data on light conditions."
        )

    @pytest.mark.parametrize("mode_case", simulation_cases)
    def test_input_check_without_temperature(self, mode_case: BaseSimulation):
        with pytest.raises(ValueError) as e_info:
            test_sim: BaseSimulation = mode_case()
            test_sim.environment.light = 4.2
            assert test_sim.environment.temperature is None
            test_sim.validate_environment()
        assert (
            str(e_info.value)
            == "CoralModel simulation cannot run without data on temperature conditions."
        )

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
    def test_validate_constants_with_valid_values(
        self, valid_value: Union[CoralConstants, str, Path]
    ):
        return_value = BaseSimulation.validate_constants(valid_value)
        assert isinstance(return_value, CoralConstants)

    def test_validate_constants_with_not_valid_value(self):
        with pytest.raises(NotImplementedError) as e_err:
            BaseSimulation.validate_constants(42)
        assert str(e_err.value) == f"Validator not available for {type(42)}"

    @pytest.mark.parametrize(
        "field_value, values",
        [
            pytest.param(
                Coral(
                    **dict(
                        constants=CoralConstants(),
                        dc=0.2,
                        hc=0.1,
                        bc=0.2,
                        tc=0.1,
                        ac=0.2,
                        Csp=1,
                    )
                ),
                None,
                id="As Object",
            ),
            pytest.param(
                dict(
                    constants=CoralConstants(), dc=0.2, hc=0.2, bc=0.2, ac=0.2, tc=0.2
                ),
                None,
                id="As dict with Constants object entry.",
            ),
            pytest.param(
                dict(dc=0.2, hc=0.2, bc=0.2, ac=0.2, tc=0.2),
                dict(constants=CoralConstants()),
                id="As dict with Constants in values dict entry.",
            ),
        ],
    )
    def test_validate_coral_with_valid_values(
        self, field_value: Union[Coral, dict], values: Optional[dict]
    ):
        return_value = BaseSimulation.validate_coral(field_value, values)
        assert isinstance(return_value, Coral)

    def test_validate_coral_with_invalid_values(self):
        with pytest.raises(NotImplementedError) as e_err:
            BaseSimulation.validate_coral(42, None)
        assert str(e_err.value) == f"Validator not available for {type(42)}"

    def test_validate_coral_with_dict_without_constants(self):
        with pytest.raises(ValueError) as e_err:
            BaseSimulation.validate_coral(
                dict(aKey="aValue"), dict(anotherKey="anotherValue")
            )
        assert (
            str(e_err.value)
            == "Constants should be provided to initialize a Coral Model."
        )

    @pytest.mark.parametrize(
        "model_type",
        [
            pytest.param(DimrModel, id="DIMR model"),
            pytest.param(FlowFmModel, id="FlowFM model"),
            pytest.param(Transect, id="Transect"),
        ],
    )
    def test_validate_hydrodynamics_present_given_hydromodel(
        self, model_type: HydrodynamicProtocol
    ):
        model_in = model_type()
        model_out = BaseSimulation.validate_hydrodynamics_present(model_in, dict())
        assert isinstance(model_out, HydrodynamicProtocol)
        assert model_in == model_out


class TestSimulation:
    @pytest.mark.parametrize("mode", HydrodynamicsFactory.supported_modes)
    def test_init_simulation(self, mode: HydrodynamicProtocol):
        test_sim = Simulation(mode=mode.__name__)
        assert isinstance(test_sim.hydrodynamics, mode)
        # Verify abstract methods raise nothing.
        test_sim.configure_hydrodynamics()
        test_sim.configure_output()
