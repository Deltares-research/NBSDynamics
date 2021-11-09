import numpy as np
import pytest

from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.hydrodynamics.reef_1d import Reef1D


class TestReef1D:
    def test_init_reef1d(self):
        test_reef = Reef1D()
        assert isinstance(test_reef, HydrodynamicProtocol)
        assert test_reef.bath is None
        assert test_reef.Hs is None
        assert test_reef.Tp is None
        assert test_reef.dx is None
        assert test_reef.working_dir is None
        assert test_reef.definition_file is None
        assert test_reef.config_file is None
        assert test_reef.water_depth is None
        # Some of the defined properties with fix values.
        assert test_reef.y_coordinates == np.array([0])
        assert test_reef.x_coordinates == None
        assert test_reef.xy_coordinates == None
        assert test_reef.vel_wave == 0
        assert test_reef.vel_curr_mn == 0
        assert test_reef.vel_curr_mx == 0
        assert test_reef.per_wav == None
        assert test_reef.water_level == 0
        assert repr(test_reef) == (
            "Reef1D(bathymetry=None, wave_height=None, wave_period=None)"
        )
        assert test_reef.settings == (
            "One-dimensional simple hydrodynamic model to simulate the "
            "hydrodynamics on a (coral) reef with the following settings:"
            "\n\tBathymetric cross-shore data : None"
            "\n\t\trange [m]  : None"
            "\n\t\tlength [m] : None"
            "\n\tSignificant wave height [m]  : None"
            "\n\tPeak wave period [s]         : None"
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

    def test_wave_length(self):
        test_ref1d = Reef1D()
        test_ref1d.Tp = 1
        test_ref1d.bath = np.array([1])
        assert test_ref1d.wave_length[0] == pytest.approx(1.56031758)

    def test_wave_number(self):
        test_ref1d = Reef1D()
        test_ref1d.Tp = 1
        test_ref1d.bath = np.array([1])
        assert test_ref1d.wave_number[0] == pytest.approx(4.02686311)

    def test_wave_frequency(self):
        test_ref1d = Reef1D()
        test_ref1d.Tp = 1
        assert test_ref1d.wave_frequency == pytest.approx(6.2831853)

    def test_wave_celerity(self):
        test_ref1d = Reef1D()
        test_ref1d.Tp = 1
        test_ref1d.bath = np.array([1])
        assert test_ref1d.wave_celerity[0] == pytest.approx(1.56031758)

    def test_group_celerity(self):
        test_ref1d = Reef1D()
        test_ref1d.Tp = 1
        test_ref1d.bath = np.array([1])
        assert test_ref1d.group_celerity[0] == pytest.approx(1.00429061)

    def test_initiate(self):
        with pytest.raises(NotImplementedError):
            Reef1D().initiate()

    @pytest.mark.parametrize(
        "storm_value, expected_result",
        [
            pytest.param(True, (None, None), id="With Storm"),
            pytest.param(False, (None, None, None), id="Without Storm"),
        ],
    )
    def test_update(self, storm_value: bool, expected_result: tuple):
        test_reef1d = Reef1D()
        assert test_reef1d.update(coral=None, storm=storm_value) == expected_result

    def test_finalise(self):
        with pytest.raises(NotImplementedError):
            Reef1D().finalise()
