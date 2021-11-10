from typing import Any
import pytest

from src.core.constants import Constants
from test.utils import TestUtils


class TestConstants:
    def test_from_input_file(self):
        test_file = (
            TestUtils.get_local_test_file("transect_case") / "input" / "coral_input.txt"
        )
        result_constants = Constants.from_input_file(test_file)
        assert result_constants is not None

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(1e6, id="As float"),
            pytest.param("1e6", id="As string"),
            pytest.param(1000000, id="As int"),
        ],
    )
    def test_validate_scientific_int_value(self, value: Any):
        assert Constants.validate_scientific_int_value(value) == 1000000

    def test_validate_scientific_int_raises(self):
        with pytest.raises(NotImplementedError) as e_err:
            Constants.validate_scientific_int_value(None)
        assert str(e_err.value) == f"No converter available for {type(None)}."

    @pytest.mark.parametrize(
        "input_dict, output_dict",
        [
            pytest.param(
                dict(pfd=False, fme=True, warn_proc=True),
                dict(pfd=False, fme=False, tme=False, warn_proc=True),
                id="PFD False, FME True, warn_proc True",
            ),
            pytest.param(
                dict(pfd=False, fme=False, warn_proc=True),
                dict(pfd=False, fme=False, tme=False, warn_proc=True),
                id="PFD False, FME False, warn_proc True",
            ),
            pytest.param(
                dict(pfd=False, fme=True, warn_proc=False),
                dict(pfd=False, fme=False, tme=False, warn_proc=False),
                id="PFD False, FME True, warn_proc False",
            ),
            pytest.param(
                dict(pfd=True, fme=True, tme=True, warn_proc=False),
                dict(pfd=True, fme=True, tme=True, warn_proc=False),
                id="PFD True, FME True, TME True, warn_proc False",
            ),
            pytest.param(
                dict(pfd=True, fme=False, tme=True, warn_proc=True),
                dict(pfd=True, fme=False, tme=False, warn_proc=True),
                id="PFD True, FME False, TME True, warn_proc True",
            ),
            pytest.param(
                dict(pfd=True, fme=False, tme=False, warn_proc=False),
                dict(pfd=True, fme=False, tme=False, warn_proc=False),
                id="PFD True, FME False, TME False, warn_proc False",
            ),
        ],
    )
    def test_check_processes(self, input_dict: dict, output_dict: dict):
        result_dict = Constants.check_processes(input_dict)
        assert result_dict == output_dict
