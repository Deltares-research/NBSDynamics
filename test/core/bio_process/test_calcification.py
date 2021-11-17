import pytest
from test.core.bio_process.bio_utils import valid_coral
from src.core.bio_process.calcification import Calcification
from src.core.constants import Constants
from src.core.coral.coral_model import Coral
import numpy as np


class TestCalcification:
    @pytest.fixture(autouse=True)
    def calc_test(self) -> Calcification:
        return Calcification(Constants())

    def test_init_calcification(self, calc_test: Calcification):
        assert calc_test.ad == 1

    def test_calcification_rate(self, calc_test: Calcification, valid_coral: Coral):
        valid_coral.pop_states = np.array([[[[1]]]])
        valid_coral.photo_rate = 1
        omegas = np.linspace(1, 5, 4)
        answer = [0.28161333, 0.38378897, 0.42082994, 0.43996532]
        for i, omega in enumerate(omegas):
            calc_test.calcification_rate(valid_coral, omega)
            assert float(valid_coral.calc), pytest.approx(answer[i])
