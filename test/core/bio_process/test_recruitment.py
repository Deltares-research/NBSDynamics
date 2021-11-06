import pytest

from src.core.bio_process.recruitment import Recruitment
from src.core.coral_model import Coral


class TestRecruitment:
    def test_spawmning_recruitment(self):
        test_recr = Recruitment(constants=None)
        test_coral = Coral(0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 1)
        with pytest.raises(ValueError) as e_info:
            test_recr.spawning(test_coral, "C")

        assert str(e_info.value) == "C not in ('P', 'V')."
