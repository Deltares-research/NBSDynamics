from test.core.bio_process.bio_utils import valid_coral

import pytest

from src.core.bio_process.recruitment import Recruitment
from src.core.coral.coral_model import Coral


class TestRecruitment:
    def test_spawmning_recruitment(self, valid_coral: Coral):
        test_recr = Recruitment(constants=None)
        with pytest.raises(ValueError) as e_info:
            test_recr.spawning(valid_coral, "C")

        assert str(e_info.value) == "C not in ('P', 'V')."
