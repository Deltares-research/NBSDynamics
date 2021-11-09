import pytest

from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.hydrodynamics.reef_0d import Reef0D


class TestReef0D:
    def test_init_reef0d(self):
        test_reef = Reef0D()
        assert isinstance(test_reef, HydrodynamicProtocol)
        assert test_reef.settings == "Not yet implemented."
        assert test_reef.x_coordinates is None
        assert test_reef.y_coordinates is None
        assert test_reef.water_depth is None
        assert test_reef.working_dir is None
        assert test_reef.definition_file is None
        assert test_reef.config_file is None
        assert test_reef.xy_coordinates is None
        assert test_reef.space is None

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
