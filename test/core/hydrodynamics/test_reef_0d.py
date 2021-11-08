import pytest

from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.hydrodynamics.reef_0d import Reef0D


class TestReef0D:
    def test_init_reef0d(self):
        test_reef = Reef0D()
        assert test_reef.settings == "Not yet implemented."
        assert test_reef.x_coordinates == None
        assert test_reef.y_coordinates == None
        assert test_reef.water_depth == None
        with pytest.raises(ValueError) as e_err:
            test_reef.xy_coordinates
        assert (
            str(e_err.value)
            == "XY Coordinates require both 'x_coorinates' and 'y_coordinates' to be defined."
        )
        with pytest.raises(ValueError) as e_err:
            test_reef.space
        assert (
            str(e_err.value)
            == "XY Coordinates require both 'x_coorinates' and 'y_coordinates' to be defined."
        )

    @pytest.mark.parametrize(
        "storm_value, expected_value",
        [
            pytest.param(True, (None, None), id="Storm is True"),
            pytest.param(False, (None, None, None), id="Storm is False"),
        ],
    )
    def test_update(self, storm_value: bool, expected_value):
        test_base = Reef0D()
        result = test_base.update(coral=None, storm=storm_value)
        assert result == expected_value

    def test_initiate(self):
        with pytest.raises(NotImplementedError):
            Reef0D().initiate()

    def test_finalise(self):
        with pytest.raises(NotImplementedError):
            Reef0D().finalise()
