import pytest

from src.core.environment import Constants
from test.utils import TestUtils


class TestConstants:
    def test_from_input_file(self):
        test_file = (
            TestUtils.get_local_test_file("transect_case") / "input" / "coral_input.txt"
        )
        result_constants = Constants.from_input_file(test_file)
        assert result_constants is not None
