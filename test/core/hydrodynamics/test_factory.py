from typing import Iterable, List, Tuple, Type

import numpy as np
import pytest

from src.core.hydrodynamics.delft3d import Delft3D
from src.core.hydrodynamics.factory import HydrodynamicsFactory
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.hydrodynamics.reef_0d import Reef0D
from src.core.hydrodynamics.reef_1d import Reef1D
from src.core.hydrodynamics.transect import Transect


def valid_model_cases() -> Iterable[pytest.param]:
    def get_all_cases(type_name: Type) -> Iterable[pytest.param]:
        name = type_name.__name__
        yield pytest.param(name, type_name, id=name)
        yield pytest.param(name.lower(), type_name, id=name.lower())
        yield pytest.param(name.upper(), type_name, id=name.upper())

    cases_from = [Reef0D, Reef1D, Delft3D, Transect]
    cases: List[Tuple[str]] = []
    for case in cases_from:
        cases.extend(get_all_cases(case))
    return cases


class TestHydrodynamicsFactory:
    @pytest.mark.parametrize(
        "mode, expected_type",
        valid_model_cases(),
    )
    def test_get_hydrodynamic_model_type(self, mode: str, expected_type: Type):
        mapped_type = HydrodynamicsFactory.get_hydrodynamic_model_type(mode)
        assert mapped_type == expected_type
        assert isinstance(
            mapped_type, HydrodynamicProtocol
        ), f"{expected_type} does not fully implement the HydrodynamicProtocol."

    @pytest.mark.parametrize(
        "unknown_mode",
        [
            pytest.param("", id="Empty string"),
            pytest.param(None, id="None as input"),
            pytest.param("another", id="Unknown type."),
        ],
    )
    def test_set_model_mode_unknown_raises_valueerror(self, unknown_mode: str):
        # 1. Set up test data.
        expected_mssg = (
            f"{unknown_mode} not in ['Reef0D', 'Reef1D', 'Delft3D', 'Transect']."
        )

        # 2. Run test.
        with pytest.raises(ValueError) as e_info:
            HydrodynamicsFactory.get_hydrodynamic_model_type(unknown_mode)

        # 3. Verify final expectation
        assert str(e_info.value) == expected_mssg
