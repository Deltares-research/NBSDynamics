from datetime import datetime
from pathlib import Path

import numpy as np
import pandas
import pytest

from src.core.output_model import Output


class TestOutput:
    def test_output_init(self):
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        outpoint_array = np.array([False, True])
        now_time = datetime.now()
        test_output = Output(
            **dict(
                output_dir=Path(),
                xy_coordinates=xy_array,
                outpoint=outpoint_array,
                first_date=now_time,
            )
        )
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
        test_output = Output(
            **dict(
                output_dir=Path(),
                xy_coordinates=xy_array,
                outpoint=outpoint_array,
                first_date=now_time,
            )
        )
        test_output.map_output = Path()
        test_output.his_output = Path()
        assert str(test_output) == "Output exported:\n\t.\n\t."
        assert test_output.defined == True

    @pytest.mark.parametrize(
        "unknown_type",
        [
            pytest.param("MAP"),
            pytest.param("HIS"),
            pytest.param("unknown"),
            pytest.param(""),
            pytest.param(None),
        ],
    )
    def test_define_output_with_unknown_output_type_raises(self, unknown_type: str):
        with pytest.raises(ValueError) as e_info:
            pandas_dt = pandas.to_datetime(np.array("2021-12-20", dtype=np.datetime64))
            test_output = Output(
                **dict(
                    output_dir=Path.cwd(),
                    xy_coordinates=np.array([[0, 1], [1, 0]]),
                    outpoint=None,
                    first_date=pandas_dt,
                )
            )
            test_output.define_output(unknown_type)
        assert str(e_info.value) == f"{unknown_type} not in ('map', 'his')."
