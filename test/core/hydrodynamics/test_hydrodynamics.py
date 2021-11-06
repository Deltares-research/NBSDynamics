from typing import Type

import pytest

from src.core.hydrodynamics.delft3d import Delft3D
from src.core.hydrodynamics.hydrodynamics import Hydrodynamics, Reef0D, Reef1D
from src.core.hydrodynamics.transect import Transect


class TestHydrodynamics:
    @pytest.mark.parametrize(
        "mode, expected_type",
        [
            pytest.param("Reef0D", Reef0D, id="Reef0D"),
            pytest.param("Reef1D", Reef1D, id="Reef1D"),
            pytest.param("Delft3D", Delft3D, id="Delft3D"),
            pytest.param("Transect", Transect, id="Transect"),
        ],
    )
    def test_init_hydrodynamics(self, mode: str, expected_type: Type):
        test_hd = Hydrodynamics(mode)
        assert test_hd.mode == mode
        assert isinstance(test_hd.model, expected_type)

    @pytest.mark.parametrize(
        "old_mode, old_type",
        [
            pytest.param("Reef0D", Reef0D, id="Reef0D"),
            pytest.param("Reef1D", Reef1D, id="Reef1D"),
            pytest.param("Delft3D", Delft3D, id="Delft3D"),
            pytest.param("Transect", Transect, id="Transect"),
        ],
    )
    @pytest.mark.parametrize(
        "new_mode, new_type",
        [
            pytest.param("Reef0D", Reef0D, id="Reef0D"),
            pytest.param("Reef1D", Reef1D, id="Reef1D"),
            pytest.param("Delft3D", Delft3D, id="Delft3D"),
            pytest.param("Transect", Transect, id="Transect"),
        ],
    )
    def test_set_model_changes_model_and_mode(
        self, old_mode: str, old_type: Type, new_mode: str, new_type: Type
    ):
        # 1. Define first condition
        test_hd = Hydrodynamics(old_mode)
        assert test_hd.mode == old_mode
        assert isinstance(test_hd.model, old_type)

        # 2. Do test.
        set_mode = test_hd.set_model(new_mode)

        # 3. Verify expectations
        assert set_mode == new_mode
        assert test_hd.mode == new_mode
        assert isinstance(test_hd.model, new_type)

    @pytest.mark.parametrize(
        "unknown_mode",
        [
            pytest.param("reef0d"),
            pytest.param("reef1d"),
            pytest.param("delft3d"),
            pytest.param("transect"),
        ],
    )
    def test_set_model_mode_lowercase_raises_valueerror(self, unknown_mode: str):
        # 1. Set up test data.
        test_hd = Hydrodynamics("Reef0D")
        expected_mssg = (
            f"{unknown_mode} not in ('Reef0D', 'Reef1D', 'Delft3D', 'Transect')."
        )

        # 2. Run test.
        with pytest.raises(ValueError) as e_info:
            test_hd.set_model(unknown_mode)

        # 3. Verify final expectation
        assert str(e_info.value) == expected_mssg

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
        test_hd = Hydrodynamics("Reef0D")
        expected_mssg = (
            f"{unknown_mode} not in ('Reef0D', 'Reef1D', 'Delft3D', 'Transect')."
        )

        # 2. Run test.
        with pytest.raises(ValueError) as e_info:
            test_hd.set_model(unknown_mode)

        # 3. Verify final expectation
        assert str(e_info.value) == expected_mssg
