import pytest
from typing import Tuple, Any

from src.core.utils import DataReshape, SpaceTime


class TestSpaceTime:
    @pytest.mark.parametrize(
        "st_value, expected", [pytest.param(None, (1, 1)), pytest.param((4, 2), (4, 2))]
    )
    def test_init_spacetime(self, st_value: Tuple[int, int], expected: Tuple[int, int]):
        test_st = SpaceTime(st_value)
        assert test_st.spacetime == expected
        assert str(test_st) == f"{expected}"
        assert test_st.space == expected[0]
        assert test_st.time == expected[1]

    def test_set_space_changes_spacetime(self):
        test_st = SpaceTime(None)
        assert test_st.spacetime == (1, 1)
        test_st.space = 42
        assert test_st.spacetime == (42, 1)

    def test_set_time_changes_spacetime(self):
        test_st = SpaceTime(None)
        assert test_st.spacetime == (1, 1)
        test_st.time = 42
        assert test_st.spacetime == (1, 42)

    @pytest.mark.parametrize(
        "st_value, expected_err",
        [
            pytest.param(
                42, "spacetime must be of type tuple, <class 'int'> is given."
            ),
            pytest.param(
                (4.2, 2.4),
                "spacetime must consist of integers only, [<class 'float'>, <class 'float'>] is given.",
            ),
        ],
    )
    def test_set_spacetime_invalid_type_raises_exception(
        self, st_value: Any, expected_err: str
    ):
        test_st = SpaceTime(None)
        with pytest.raises(TypeError) as e_info:
            test_st.spacetime = st_value
        assert str(e_info.value) == expected_err

    def test_set_spacetime_invalid_value_raises_exception(self):
        test_st = SpaceTime(None)
        with pytest.raises(ValueError) as e_info:
            test_st.spacetime = (1, 2, 3)
        assert str(e_info.value) == "spacetime must be of size 2, 3 is given."


class TestDataReshape:
    def test_variable2arra_raises_with_str(self):
        with pytest.raises(NotImplementedError) as e_info:
            DataReshape.variable2array("aString")

        assert str(e_info.value) == "Variable cannot be of <class 'str'>."
