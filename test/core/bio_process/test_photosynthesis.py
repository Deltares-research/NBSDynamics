from test.core.bio_process.bio_utils import (
    coral_2x2,
    matrix_1x1,
    matrix_2x2,
    valid_coral,
)

import numpy as np
import pytest

from src.core.bio_process.photosynthesis import Photosynthesis
from src.core.common.constants import Constants
from src.core.common.space_time import DataReshape
from src.core.coral.coral_model import Coral


class TestPhotosynthesis:
    def test_init_photoshynthesis(self, matrix_1x1: DataReshape):
        assert matrix_1x1.spacetime == (1, 1)
        input_dict = dict(constants=None, light_in=None, first_year=None)
        test_photo = Photosynthesis(**input_dict)
        assert test_photo.pld == 1
        assert test_photo.ptd == 1
        assert test_photo.pfd == 1
        assert test_photo.constants == None

    @pytest.fixture(autouse=False)
    def valid_photosynthesis(self, matrix_1x1: DataReshape) -> Photosynthesis:
        class TestConstants:
            pfd = 1
            pfd_min = 0
            ucr = 2

        assert matrix_1x1.spacetime == (1, 1)
        input_dict = dict(
            constants=TestConstants(),
            light_in=None,
            first_year=None,
        )
        return Photosynthesis(**input_dict)

    def test_flow_dependency_no_pfd(self, valid_photosynthesis: Photosynthesis):
        valid_photosynthesis.constants.pfd = None
        valid_photosynthesis.flow_dependency(None)
        assert valid_photosynthesis.pfd == 1

    def test_flow_dependency_with_pfd(
        self, valid_photosynthesis: Photosynthesis, valid_coral: Coral
    ):
        assert valid_photosynthesis.pfd == 1
        valid_coral.ucm = 1
        valid_photosynthesis.flow_dependency(valid_coral)
        assert valid_photosynthesis.pfd[0][0] == pytest.approx(0.76159416)

    def test_light_dependency_raises_notimplemented_error(
        self, valid_photosynthesis: Photosynthesis, valid_coral: Coral
    ):
        with pytest.raises(NotImplementedError) as e_info:
            valid_photosynthesis.light_dependency(valid_coral, None)
        assert (
            str(e_info.value)
            == "Only the quasi-steady state solution is currently implemented; use key-word 'qss'."
        )

    @pytest.fixture(autouse=False)
    def photo_legacy(self, matrix_1x1: DataReshape) -> Photosynthesis:
        assert matrix_1x1.spacetime == (1, 1)
        return Photosynthesis(Constants(), 600, False)

    def test_photosynthetic_light_dependency(
        self, photo_legacy: Photosynthesis, valid_coral: Coral
    ):
        valid_coral.light = 600
        photo_legacy.light_dependency(valid_coral, "qss")
        assert float(photo_legacy.pld), pytest.approx(0.90727011)

    def test_photosynthetic_flow_dependency(
        self, photo_legacy: Photosynthesis, valid_coral: Coral
    ):
        photo_legacy.constants = valid_coral.constants
        valid_coral.ucm = 0.1
        photo_legacy.pfd = True
        photo_legacy.flow_dependency(valid_coral)
        assert float(photo_legacy.pfd), pytest.approx(0.94485915)
        photo_legacy.pfd = False
        photo_legacy.flow_dependency(valid_coral)
        assert float(photo_legacy.pfd) == 1


class TestPhotosynthesis2x2:
    """
    Legacy tests with a DataReshape 2x2 matrix.
    """

    @pytest.fixture(autouse=False)
    def photo_2x2(self, matrix_2x2: DataReshape) -> Photosynthesis:
        assert matrix_2x2.spacetime == (2, 2)
        return Photosynthesis(Constants(), [600, 600], False)

    def test_initiation(self, photo_2x2: Photosynthesis, matrix_2x2: DataReshape):
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(photo_2x2.I0[i, j]) == 600
        assert photo_2x2.first_year is False
        assert float(photo_2x2.pld) == 1
        assert float(photo_2x2.ptd) == 1
        assert float(photo_2x2.pfd) == 1

    def test_photosynthetic_light_dependency(
        self, photo_2x2: Photosynthesis, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        coral_2x2.initiate_coral_morphology()
        coral_2x2.light = [600, 600]
        photo_2x2.light_dependency(coral_2x2, "qss")
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(photo_2x2.pld[i, j]), pytest.approx(0.90727011)

    def test_photosynthetic_flow_dependency(
        self, photo_2x2: Photosynthesis, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        photo_2x2.constants = coral_2x2.constants
        coral_2x2.initiate_coral_morphology()
        coral_2x2.ucm = matrix_2x2.variable2array([0.1, 0.1])
        coral_2x2.constants.pfd = True
        photo_2x2.flow_dependency(coral_2x2)
        for i in range(matrix_2x2.space):
            for j in range(matrix_2x2.time):
                assert float(photo_2x2.pfd[i, j]), pytest.approx(0.94485915)
        coral_2x2.constants.pfd = False
        photo_2x2.flow_dependency(coral_2x2)
        assert float(photo_2x2.pfd) == 1
