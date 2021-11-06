from typing import Type

import pytest

from src.core.hydrodynamics.delft3d import Delft3D
from src.core.hydrodynamics.hydrodynamics import (
    BaseHydro,
    Hydrodynamics,
    Reef0D,
    Reef1D,
)
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
        assert (
            str(test_hd)
            == f"Coupled hydrodynamic model: {str(test_hd.model)}\n\tmode={mode}"
        )
        assert repr(test_hd) == f"Hydrodynamics(mode={mode})"

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


class TestBaseHydro:
    def test_init_base_hydro(self) -> BaseHydro:
        test_base = BaseHydro()
        assert test_base.update_interval_storm == None
        assert test_base.update_interval == None
        assert test_base.settings == "No hydrodynamic model coupled."
        assert test_base.x == None
        assert test_base.y == None

    @pytest.mark.parametrize(
        "storm_value, expected_value",
        [
            pytest.param(True, (None, None), id="Storm is True"),
            pytest.param(False, (None, None, None), id="Storm is False"),
        ],
    )
    def test_update(self, storm_value: bool, expected_value):
        test_base = BaseHydro()
        result = test_base.update(coral=None, storm=storm_value)
        assert result == expected_value


class TestReef0D:
    def test_init_reef0d(self):
        test_reef = Reef0D()
        assert isinstance(test_reef, BaseHydro)
        assert test_reef.settings == "Not yet implemented."


class TestReef1D:
    def test_init_reef1d(self):
        test_reef = Reef1D()
        assert isinstance(test_reef, BaseHydro)
        assert test_reef.bath == None
        assert test_reef.Hs == None
        assert test_reef.Tp == None
        assert test_reef.dx == None
        # Some of the defined properties with fix values.
        assert test_reef.y == 0
        assert test_reef.vel_wave == 0
        assert test_reef.vel_curr_mn == 0
        assert test_reef.vel_curr_mx == 0
        assert test_reef.per_wav == None
        assert test_reef.water_level == 0
        assert repr(test_reef) == (
            "Reef1D(bathymetry=None, wave_height=None, wave_period=None)"
        )

    def test_settings(self):
        test_reef = Reef1D()
        test_reef.bath = [4]
        test_reef.dx = 1
        assert test_reef.settings == (
            "One-dimensional simple hydrodynamic model to simulate the "
            "hydrodynamics on a (coral) reef with the following settings:"
            "\n\tBathymetric cross-shore data : list"
            "\n\t\trange [m]  : 4-4"
            "\n\t\tlength [m] : 1"
            "\n\tSignificant wave height [m]  : None"
            "\n\tPeak wave period [s]         : None"
        )

    @pytest.fixture(autouse=True)
    def reef_1d(self) -> Reef1D:
        """
        Initializes a valid Reef1D to be used in the tests.

        Returns:
            Reef1D: Valid Reef1D for testing.
        """
        return Reef1D()

    def test_set_can_dia(self, reef_1d: Reef1D):
        reef_1d.can_dia = 4.2
        assert reef_1d.can_dia == 4.2

    def test_set_can_height(self, reef_1d: Reef1D):
        reef_1d.can_height = 4.2
        assert reef_1d.can_height == 4.2

    def test_set_can_den(self, reef_1d: Reef1D):
        reef_1d.can_den = 4.2
        assert reef_1d.can_den == 4.2

    def test_dispersion(self):
        val = Reef1D.dispersion(4, 2, 2, 4)
        assert val == pytest.approx(1.463013990480674)
