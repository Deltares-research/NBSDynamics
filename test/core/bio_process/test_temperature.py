from src.core.bio_process.temperature import Temperature
import pytest


class TestTemperature:
    def test_coral_temperature(self):
        class TestCoral:
            temp = 42
            delta_t = 2.4
            light = 4.2

        class TestConstants:
            tme = 4.2
            ap = 1
            k = 2
            K0 = 3

        test_temp = Temperature(TestConstants(), 42)
        test_coral = TestCoral()
        test_temp.coral_temperature(test_coral)
        assert test_coral.temp[0][0] == pytest.approx(43.68)
