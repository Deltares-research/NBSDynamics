from test.core.bio_process.bio_utils import valid_coral

from src.core.bio_process.temperature import Temperature
from src.core.coral.coral_model import Coral
from src.core.utils import DataReshape


class TestTemperature:
    def test_coral_temperature(self, valid_coral: Coral):
        class TestConstants:
            tme = 4.2
            ap = 1
            k = 2
            K0 = 3

        test_temp = Temperature(TestConstants(), 42, DataReshape())
        valid_coral.temp = 42
        valid_coral.delta_t = 2.4
        valid_coral.light = 4.2
        test_temp.coral_temperature(valid_coral)
        assert valid_coral.temp[0][0] == 43.68
