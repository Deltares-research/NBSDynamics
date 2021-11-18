from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

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

    def test_default(self):
        spacetime = SpaceTime().spacetime
        assert len(spacetime) == 2
        assert isinstance(spacetime, tuple)

    def test_default_input(self):
        spacetime = SpaceTime(None).spacetime
        assert len(spacetime) == 2
        assert isinstance(spacetime, tuple)

    @pytest.mark.parametrize("value", [(int(1)), (float(1)), (str(1))])
    def test_global_raise_type_error(self, value: Any):
        with pytest.raises(TypeError):
            SpaceTime(value)

    @pytest.mark.parametrize("value", [((1, 1)), ([1, 1])])
    def test_global_not_raise_type_error(self, value):
        SpaceTime(value)

    @pytest.mark.parametrize("value", [((1,)), ((1, 1, 1))])
    def test_size_error(self, value: tuple):
        with pytest.raises(ValueError):
            SpaceTime(value)

    @pytest.mark.parametrize("value", [((float(1), 1))])
    def test_local_raise_type_error(self, value):
        with pytest.raises(TypeError):
            SpaceTime(value)

    def test_return_type(self):
        assert isinstance(SpaceTime((1, 1)).spacetime, tuple)
        assert isinstance(SpaceTime([1, 1]).spacetime, tuple)


class TestDataReshape:
    def test_variable2arra_raises_with_str(self):
        with pytest.raises(NotImplementedError) as e_info:
            DataReshape.variable2array("aString")

        assert str(e_info.value) == "Variable cannot be of <class 'str'>."

    def test_default_spacetime(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        assert isinstance(reshape.spacetime, tuple)
        assert reshape.spacetime == spacetime
        assert isinstance(reshape.space, int)
        assert reshape.space == spacetime[0]
        assert isinstance(reshape.time, int)
        assert reshape.time == spacetime[1]

    @pytest.mark.parametrize("value", [(int(1)), (float(1)), (str(1))])
    def test_set_spacetime_raise_type_error(self, value: Any):
        with pytest.raises(TypeError):
            DataReshape(value)

    @pytest.mark.parametrize("value", [((1, 1)), ([1, 1])])
    def test_set_spacetime_not_raise_error(self, value: Any):
        DataReshape(value)

    def test_variable2array(self):
        assert isinstance(DataReshape.variable2array(float(1)), np.ndarray)
        assert isinstance(DataReshape.variable2array(int(1)), np.ndarray)
        with pytest.raises(NotImplementedError):
            DataReshape.variable2array(str(1))
        assert isinstance(DataReshape.variable2array((1, 1)), np.ndarray)
        assert isinstance(DataReshape.variable2array([1, 1]), np.ndarray)

    @pytest.mark.parametrize(
        "dimension, values",
        [
            pytest.param("space", [0, 1, 2, 4], id="Space"),
            pytest.param("time", [0, 1, 2, 4, 8], id="Time"),
        ],
    )
    def test_variable2matrix_shape(self, dimension: str, values: List[float]):
        reshape = DataReshape((4, 5))
        matrix = reshape.variable2matrix(values, dimension)
        assert matrix.shape == (4, 5)

    def test_variable2matrix_value_space(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        matrix = reshape.variable2matrix(var, "space")
        for i, row in enumerate(matrix):
            for col in row:
                assert col == var[i]

    def test_variable2matrix_value_time(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        matrix = reshape.variable2matrix(var, "time")
        for row in matrix:
            for i, col in enumerate(row):
                assert col == var[i]

    @pytest.mark.parametrize(
        "variable, values",
        [
            pytest.param("space", [0, 1, 2, 4, 8], id="Space"),
            pytest.param("time", [0, 1, 2, 4], id="Time"),
        ],
    )
    def test_variable2_matrix_raises_valueerror(
        self, variable: str, values: List[float]
    ):
        reshape = DataReshape((4, 5))
        with pytest.raises(ValueError):
            reshape.variable2matrix(values, variable)

    @pytest.mark.parametrize(
        "dimension, conversion, expected_result",
        [
            pytest.param("space", None, [8, 8, 8, 8], id="Space-None"),
            pytest.param("space", "mean", [3, 3, 3, 3], id="Space-Mean"),
            pytest.param("space", "max", [8, 8, 8, 8], id="Space-Max"),
            pytest.param("space", "min", [0, 0, 0, 0], id="Space-Min"),
            pytest.param("space", "sum", [15, 15, 15, 15], id="Space-Sum"),
            pytest.param("time", None, [0, 1, 2, 4, 8], id="Time-None"),
            pytest.param("time", "min", [0, 1, 2, 4, 8], id="Time-Min"),
            pytest.param("time", "max", [0, 1, 2, 4, 8], id="Time-Max"),
            pytest.param("time", "mean", [0, 1, 2, 4, 8], id="Time-Mean"),
            pytest.param("time", "sum", [0, 4, 8, 16, 32], id="Time-Sum"),
        ],
    )
    def test_matrix2array(
        self, dimension: str, conversion: Optional[str], expected_result: List[float]
    ):
        reshape = DataReshape((4, 5))
        var = np.array(
            [
                [0, 1, 2, 4, 8],
                [0, 1, 2, 4, 8],
                [0, 1, 2, 4, 8],
                [0, 1, 2, 4, 8],
            ]
        )
        result = reshape.matrix2array(var, dimension, conversion)
        for i, val in enumerate(expected_result):
            assert result[i] == val
