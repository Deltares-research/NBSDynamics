from datetime import datetime
import pytest
from src.core.output_model import Output
from pathlib import Path
import numpy as np


class TestOutput:
    def test_output_init(self):
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        outpoint_array = np.array([False, True])
        now_time = datetime.now()
        test_output = Output(Path(), xy_array, outpoint_array, now_time)
        assert str(test_output) == "Output undefined."
        assert (
            repr(test_output)
            == f"Output(xy_coordinates=[[0. 1.]\n [1. 0.]], first_date={now_time})"
        )
        assert test_output.first_year == now_time.year
        assert test_output.first_date == now_time
        assert test_output.space == 2
        assert test_output.defined == False

    def test_output_set_map_output_and_his(self):
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        outpoint_array = np.array([False, True])
        now_time = datetime.now()
        test_output = Output(Path(), xy_array, outpoint_array, now_time)
        test_output._map_output = Path()
        test_output._his_output = Path()
        assert str(test_output) == "Output exported:\n\t.\n\t."
        assert test_output.defined == True
