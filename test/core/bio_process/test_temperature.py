from test.core.bio_process.bio_utils import valid_coral

import pytest

from src.core.bio_process.temperature import Temperature
from src.core.constants import Constants
from src.core.coral.coral_model import Coral
from src.core.utils import DataReshape


class TestTemperature:
    @pytest.fixture(autouse=False)
    def temp_test(self) -> Temperature:
        return Temperature(Constants(), 300, DataReshape())

    def test_initiation(self, temp_test: Temperature):
        assert temp_test.T == 300

    def test_coral_temperature(self, temp_test: Temperature, valid_coral: Coral):
        valid_coral.delta_t = 0.001
        valid_coral.light = 600
        temp_test.coral_temperature(valid_coral)
        assert float(valid_coral.temp), pytest.approx(300.00492692)

    def test_no_tme(self, temp_test: Temperature, valid_coral: Coral):
        # core.PROCESSES.tme = False
        valid_coral.delta_t = 0.001
        valid_coral.light = 600
        temp_test.coral_temperature(valid_coral)
        assert float(valid_coral.temp), pytest.approx(300)
