from src.core.bio_process.temperature import Temperature
from src.core.coral.coral_model import Coral
from src.core.utils import DataReshape


class TestTemperature:
    def test_coral_temperature(self):
        class TestConstants:
            tme = 4.2
            ap = 1
            k = 2
            K0 = 3

        test_temp = Temperature(TestConstants(), 42, DataReshape())
        test_coral = Coral(TestConstants(), 1, 1, 1, 1, 1, 1)
        test_coral.temp = 42
        test_coral.delta_t = 2.4
        test_coral.light = 4.2
        test_temp.coral_temperature(test_coral)
        assert test_coral.temp[0][0] == 43.68
