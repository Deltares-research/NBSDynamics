from test.core.bio_process.bio_utils import valid_coral, coral_2x2

import numpy as np
import pytest

from src.core.bio_process.photosynthesis import Photosynthesis
from src.core.coral.coral_model import Coral
from src.core.utils import DataReshape
from src.core.constants import Constants


class TestPhotosynthesis:
    def test_init_photoshynthesis(self):
        input_dict = dict(
            constants=None, light_in=None, first_year=None, datareshape=DataReshape()
        )
        test_photo = Photosynthesis(**input_dict)
        assert test_photo.pld == 1
        assert test_photo.ptd == 1
        assert test_photo.pfd == 1
        assert test_photo.constants == None

    @pytest.fixture(autouse=False)
    def valid_photosynthesis(self) -> Photosynthesis:
        class TestConstants:
            pfd = 1
            pfd_min = 0
            ucr = 2

        input_dict = dict(
            constants=TestConstants(),
            light_in=None,
            first_year=None,
            datareshape=DataReshape(),
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
    def photo_legacy(self) -> Photosynthesis:
        return Photosynthesis(Constants(), 600, False, DataReshape())

    def test_photosynthetic_light_dependency(
        self, photo_legacy: Photosynthesis, valid_coral: Coral
    ):
        valid_coral.light = 600
        photo_legacy.light_dependency(valid_coral, "qss")
        assert float(photo_legacy.pld), pytest.approx(0.90727011)

    def test_photosynthetic_flow_dependency(
        self, photo_legacy: Photosynthesis, valid_coral: Coral
    ):
        valid_coral.ucm = 0.1
        photo_legacy.flow_dependency(valid_coral)
        assert float(photo_legacy.pfd), pytest.approx(0.94485915)
        # core.PROCESSES.pfd = False
        photo_legacy.flow_dependency(valid_coral)
        assert float(photo_legacy.pfd) == 1


class TestPhotosynthesis2x2:
    """
    Legacy tests with a DataReshape 2x2 matrix.
    """

    @pytest.fixture(autouse=False)
    def photo_2x2(self) -> Photosynthesis:
        return Photosynthesis(Constants(), [600, 600], False, DataReshape((2, 2)))

    def test_initiation(self, photo_2x2: Photosynthesis):
        for i in range(photo_2x2.datareshape.space):
            for j in range(photo_2x2.datareshape.time):
                assert float(photo_2x2.I0[i, j]) == 600
        assert photo_2x2.first_year is False
        assert float(photo_2x2.pld) == 1
        assert float(photo_2x2.ptd) == 1
        assert float(photo_2x2.pfd) == 1

    def test_photosynthetic_light_dependency(
        self, photo_2x2: Photosynthesis, coral_2x2: Coral
    ):
        coral_2x2.initiate_coral_morphology()
        coral_2x2.light = [600, 600]
        photo_2x2.light_dependency(coral_2x2, "qss")
        for i in range(coral_2x2.RESHAPE.space):
            for j in range(coral_2x2.RESHAPE.time):
                assert float(photo_2x2.pld[i, j]), pytest.approx(0.90727011)

    def test_photosynthetic_flow_dependency(
        self, photo_2x2: Photosynthesis, coral_2x2: Coral
    ):
        photo_2x2.constants = coral_2x2.constants
        coral_2x2.initiate_coral_morphology()
        coral_2x2.ucm = coral_2x2.RESHAPE.variable2array([0.1, 0.1])
        coral_2x2.constants.pfd = True
        photo_2x2.flow_dependency(coral_2x2)
        for i in range(coral_2x2.RESHAPE.space):
            for j in range(coral_2x2.RESHAPE.time):
                assert float(photo_2x2.pfd[i, j]), pytest.approx(0.94485915)
        coral_2x2.constants.pfd = False
        photo_2x2.flow_dependency(coral_2x2)
        assert float(photo_2x2.pfd) == 1
