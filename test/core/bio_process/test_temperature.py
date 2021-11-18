from test.core.bio_process.bio_utils import (
    coral_2x2,
    matrix_1x1,
    matrix_2x2,
    valid_coral,
)

import pytest

from src.core.bio_process.temperature import Temperature
from src.core.common.constants import _Constants as Constants
from src.core.common.space_time import DataReshape
from src.core.coral.coral_model import Coral


class TestTemperature:
    @pytest.fixture(autouse=False)
    def temp_test(self, matrix_1x1: DataReshape) -> Temperature:
        assert matrix_1x1.spacetime == (1, 1)
        return Temperature(Constants(), 300)

    def test_initiation(self, temp_test: Temperature):
        assert temp_test.T == 300

    def test_coral_temperature(self, temp_test: Temperature, valid_coral: Coral):
        valid_coral.delta_t = 0.001
        valid_coral.light = 600
        temp_test.coral_temperature(valid_coral)
        assert float(valid_coral.temp), pytest.approx(300.00492692)

    def test_no_tme(self, temp_test: Temperature, valid_coral: Coral):
        temp_test.constants = valid_coral.constants
        temp_test.constants.tme = False
        valid_coral.delta_t = 0.001
        valid_coral.light = 600
        temp_test.coral_temperature(valid_coral)
        assert float(valid_coral.temp), pytest.approx(300)


class TestTemperature2x2:
    """
    Legacy tests with a 2x2 DataReshape matrix.
    """

    @pytest.fixture(autouse=False)
    def temp_2x2(self, matrix_2x2: DataReshape) -> Temperature:
        assert matrix_2x2.spacetime == (2, 2)
        return Temperature(Constants(), [300, 300])

    def test_initiation(self, temp_2x2: Temperature, matrix_2x2: DataReshape):
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert temp_2x2.T[i, j] == 300

    def test_coral_temperature(
        self, temp_2x2: Temperature, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        coral_2x2.initiate_coral_morphology()
        coral_2x2.delta_t = [0.001, 0.001]
        coral_2x2.light = [600, 600]
        temp_2x2.coral_temperature(coral_2x2)
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(coral_2x2.temp[i, j]), pytest.approx(300.00492692)

    def test_no_tme(
        self, temp_2x2: Temperature, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        coral_2x2.constants.tme = False
        coral_2x2.initiate_coral_morphology()
        coral_2x2.delta_t = [0.001, 0.001]
        coral_2x2.light = [600, 600]
        temp_2x2.coral_temperature(coral_2x2)
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(coral_2x2.temp[i, j]), pytest.approx(300)
