from test.core.bio_process.bio_utils import valid_coral

import numpy as np
import pytest

from src.core.bio_process.recruitment import Recruitment
from src.core.constants import Constants
from src.core.coral.coral_model import Coral


class TestRecruitment:
    def test_init_recruitment(self):
        test_recr = Recruitment()
        assert isinstance(test_recr.constants, Constants)

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
