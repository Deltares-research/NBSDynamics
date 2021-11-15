import numpy as np
import pytest

from src.core.bio_process.photosynthesis import Photosynthesis
from src.core.coral.coral_model import Coral
from src.core.utils import DataReshape


class TestPhotosynthesis:
    def test_ini_photoshynthesis(self):
        input_dict = dict(
            constants=None, light_in=None, first_year=None, datareshape=DataReshape()
        )
        test_photo = Photosynthesis(**input_dict)
        assert test_photo.pld == 1
        assert test_photo.ptd == 1
        assert test_photo.pfd == 1
        assert test_photo.constants == None

    @pytest.fixture(autouse=True)
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

    def test_flow_dependency_with_pfd(self, valid_photosynthesis: Photosynthesis):
        assert valid_photosynthesis.pfd == 1

        class TestConstants:
            pfd = 1
            pfd_min = 1
            ucr = 1

        test_coral = Coral(TestConstants(), 1, 1, 1, 1, 1, 1)
        test_coral.ucm = 1
        valid_photosynthesis.flow_dependency(test_coral)
        assert valid_photosynthesis.pfd[0][0] == pytest.approx(0.76159416)

    def test_light_dependency_raises_notimplemented_error(
        self, valid_photosynthesis: Photosynthesis
    ):
        class constant_test:
            c1 = 1

        with pytest.raises(NotImplementedError) as e_info:
            valid_photosynthesis.light_dependency(
                Coral(constant_test(), 1, 1, 1, 1, 1, 1), None
            )
        assert (
            str(e_info.value)
            == "Only the quasi-steady state solution is currently implemented; use key-word 'qss'."
        )
