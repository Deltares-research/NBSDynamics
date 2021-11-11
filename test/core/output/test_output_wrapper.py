from datetime import datetime
from pathlib import Path

import numpy as np
from src.core.output.output_model import HisOutput, MapOutput

from src.core.output.output_wrapper import OutputWrapper


class TestOutputWrapper:
    def test_output_init(self):
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        outpoint_array = np.array([False, True])
        now_time = datetime.now()
        test_output = OutputWrapper(
            output_dir=Path.cwd(),
            xy_coordinates=xy_array,
            first_date=now_time,
            outpoint=outpoint_array,
        )

        # Verify mandatory fields were set as expected.
        assert (test_output.xy_coordinates == xy_array).all()
        assert test_output.first_date == now_time
        assert test_output.defined is False

        # Verify output models were generated.
        assert isinstance(test_output.map_output, MapOutput)
        assert isinstance(test_output.his_output, HisOutput)

        # Verify built-in fields.
        assert str(test_output) == "Output undefined."
        assert (
            repr(test_output)
            == f"Output(xy_coordinates=[[0. 1.]\n [1. 0.]], first_date={now_time})"
        )
