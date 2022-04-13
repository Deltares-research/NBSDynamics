from test.core.bio_process.bio_utils import coral_2x2, matrix_2x2, valid_coral

import numpy as np
import pytest

from src.core.bio_process.recruitment import Recruitment
from src.coral.model.coral_model import Coral
from src.core.common.base_constants import BaseConstants
from src.core.common.coral_constants import CoralConstants
from src.core.common.space_time import DataReshape


class TestRecruitment:
    def test_init_recruitment(self):
        test_recr = Recruitment()
        assert isinstance(test_recr.constants, CoralConstants)
        assert isinstance(test_recr.constants, BaseConstants)

    def test_spawmning_recruitment(self, valid_coral: Coral):
        test_recr = Recruitment()
        with pytest.raises(ValueError) as e_info:
            test_recr.spawning(valid_coral, "C")

        assert str(e_info.value) == "C not in ('P', 'V')."

    def test_spawning_cover1(self, valid_coral: Coral):
        recruitment = Recruitment()
        valid_coral.pop_states = np.array([[[1, 0, 0, 0]]])
        result = recruitment.spawning(valid_coral, "P")
        assert float(result) == 0

    def test_spawning_cover2(self, valid_coral: Coral):
        recruitment = Recruitment()
        valid_coral.pop_states = np.array([[[0.5, 0, 0, 0]]])
        result = recruitment.spawning(valid_coral, "P")
        assert float(result), pytest.approx(2.5e-5)


class TestRecruitment2x2:
    """
    Legacy tests with DataReshape 2x2 matrix.
    """

    def test_spawning_cover1(self, coral_2x2: Coral, matrix_2x2: DataReshape):
        recruitment = Recruitment()
        coral_2x2.initiate_coral_morphology()
        coral_2x2.pop_states = np.array(
            [
                [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                ],
                [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                ],
            ]
        )
        result = recruitment.spawning(coral_2x2, "P")
        for i in range(matrix_2x2.space):
            assert float(result[i]) == 0

    def test_spawning_cover2(self, coral_2x2: Coral, matrix_2x2: DataReshape):
        recruitment = Recruitment()
        coral_2x2.initiate_coral_morphology()
        coral_2x2.pop_states = np.array(
            [
                [
                    [0.5, 0, 0, 0],
                    [0.5, 0, 0, 0],
                ],
                [
                    [0.5, 0, 0, 0],
                    [0.5, 0, 0, 0],
                ],
            ]
        )
        result = recruitment.spawning(coral_2x2, "P")
        for i in range(matrix_2x2.space):
            assert float(result[i]), pytest.approx(2.5e-5)
