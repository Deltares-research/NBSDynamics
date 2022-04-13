from test.biota_models.coral.bio_process.bio_utils import (
    coral_2x2,
    matrix_1x1,
    matrix_2x2,
    no_base_coral_2x2,
    valid_coral,
)

import numpy as np
import pytest

from src.biota_models.coral.model.coral_constants import CoralConstants
from src.biota_models.coral.model.coral_model import Coral
from src.biota_models.coral.bio_process.light import Light
from src.core.common.space_time import DataReshape

tolerance = 0.0000001


class TestLight:
    @pytest.fixture(autouse=False)
    def light_test(self, matrix_1x1: DataReshape) -> Light:
        assert matrix_1x1.spacetime == (1, 1)
        return Light(600, 0.1, 5)

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
            CoralConstants().theta_max = theta
            result = light_test.side_correction(valid_coral)
            assert result < 1


class TestLight2x2:
    """
    Legacy tests with a DataReshape matrix of 2x2
    """

    @pytest.fixture(autouse=False)
    def light_2x2(self) -> Light:
        return Light([600, 600], [0.1, 0.1], [5, 5])

    def test_initiation(self, light_2x2: Light, matrix_2x2: DataReshape):
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(light_2x2.I0[i, j]) == 600
                assert float(light_2x2.Kd[i, j]) == 0.1
                assert float(light_2x2.h[i, j]) == 5

    def test_representative_light(
        self, light_2x2: Light, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        # base light
        coral_2x2.initiate_coral_morphology()
        light_2x2.rep_light(coral_2x2)
        answer = 217.0490558
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(coral_2x2.light[i, j]), pytest.approx(answer)

    def test_representative_light_no_base(
        self, light_2x2: Light, no_base_coral_2x2: Coral
    ):
        no_base_coral_2x2.initiate_coral_morphology()
        light_2x2.rep_light(no_base_coral_2x2)
        answer = 253.8318634
        for i in range(no_base_coral_2x2.RESHAPE.space):
            for j in range(no_base_coral_2x2.RESHAPE.time):
                assert float(no_base_coral_2x2.light[i, j]), pytest.approx(answer)

    def test_coral_biomass(
        self, light_2x2: Light, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        coral_2x2.initiate_coral_morphology()
        light_2x2.biomass(coral_2x2)
        answer = 0.14287642
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(coral_2x2.light_bc[i, j]), pytest.approx(answer)

    def test_coral_biomass_no_base(
        self, light_2x2: Light, no_base_coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        # no base light
        no_base_coral_2x2.initiate_coral_morphology()
        light_2x2.biomass(no_base_coral_2x2)
        answer = 0.314159265
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(no_base_coral_2x2.light_bc[i, j]), pytest.approx(answer)

    def test_base_light(
        self, light_2x2: Light, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        # base light
        coral_2x2.initiate_coral_morphology()
        result = light_2x2.base_light(coral_2x2)
        answer = 0.05478977
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(result[i, j]), pytest.approx(answer)

    def test_base_light_no_base(
        self, light_2x2: Light, no_base_coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        # no base light
        no_base_coral_2x2.initiate_coral_morphology()
        result = light_2x2.base_light(no_base_coral_2x2)
        answer: float = 0.0
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(result[i, j]) == float(answer)
