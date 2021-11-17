from test.core.bio_process.bio_utils import valid_coral

import numpy as np
import pytest

from src.core.bio_process.light import Light
from src.core.constants import Constants
from src.core.coral.coral_model import Coral
from src.core.utils import DataReshape

tolerance = 0.0000001


class TestLight:
    @pytest.fixture(autouse=False)
    def light_test(self) -> Light:
        return Light(Constants(), 600, 0.1, 5, DataReshape())

    def test_initiation(self, light_test: Light):
        assert light_test.I0, pytest.approx(600, tolerance)
        assert light_test.Kd, pytest.approx(0.1, tolerance)
        assert light_test.h, pytest.approx(5, tolerance)

    def test_representative_light(self, light_test: Light, valid_coral: Coral):
        light_test.rep_light(valid_coral)
        answer = 217.0490558
        assert float(valid_coral.light), pytest.approx(answer)

    def test_representative_light_without_base(
        self, light_test: Light, valid_coral: Coral
    ):
        # no base light
        light_test.rep_light(valid_coral)
        answer = 253.8318634
        assert float(valid_coral.light), pytest.approx(answer, tolerance)

    def test_coral_biomass(self, light_test: Light, valid_coral: Coral):
        light_test.biomass(valid_coral)
        answer = 0.14287642
        assert float(valid_coral.light_bc), pytest.approx(answer, tolerance)

    def test_coral_biomass_without_base(self, light_test: Light, valid_coral: Coral):
        light_test.biomass(valid_coral)
        answer = 0.314159265
        assert float(valid_coral.light_bc), pytest.approx(answer)

    def test_base_light(self, light_test: Light, valid_coral: Coral):
        result = light_test.base_light(valid_coral)
        answer = 0.05478977
        assert float(result), pytest.approx(answer)

    def test_base_light_without_base(self, light_test: Light, valid_coral: Coral):
        # no base light
        result = light_test.base_light(valid_coral)
        answer = 0
        assert float(result), pytest.approx(answer)

    def test_side_correction(self, light_test: Light, valid_coral: Coral):
        max_thetas = np.linspace(0, np.pi)
        for theta in max_thetas:
            Constants().theta_max = theta
            result = light_test.side_correction(valid_coral)
            assert result < 1
