from datetime import datetime
from test.utils import TestUtils

import numpy as np

from src.coral.output.coral_output_model import (
    CoralHisOutput,
    CoralMapOutput,
    CoralOutputParameters,
    _CoralOutput,
)
from src.core.output.base_output_model import BaseOutput
from src.core.output.output_protocol import OutputProtocol


class TestModelParameters:
    def test_init_modelattributes(self):
        test_params = CoralOutputParameters()
        assert test_params.lme is True
        assert test_params.fme is True
        assert test_params.tme is True
        assert test_params.pd is True
        assert test_params.ps is True
        assert test_params.calc is True
        assert test_params.md is True
        assert test_params.valid_output()


class TestCoralMapOutput:
    def test_init_mapoutput(self):
        test_dir = TestUtils.get_local_test_data_dir("BaseOutput")
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        test_map_output = CoralMapOutput(
            output_dir=test_dir, xy_coordinates=xy_array, first_year=2021
        )
        assert isinstance(test_map_output, OutputProtocol)
        assert isinstance(test_map_output, _CoralOutput)
        assert isinstance(test_map_output, BaseOutput)
        assert isinstance(test_map_output.output_params, CoralOutputParameters)

        assert test_map_output.output_filepath == test_dir / "CoralModel_map.nc"
        assert (test_map_output.xy_coordinates == xy_array).all()
        assert test_map_output.first_year == 2021
        assert test_map_output.space == len(xy_array)


class TestCoralHisOutput:
    def test_init_mapoutput(self):
        test_dir = TestUtils.get_local_test_data_dir("BaseOutput")
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        idx_array = np.array([[1, 0], [0, 1]], np.float64)
        first_date = datetime.now()
        test_his_output = CoralHisOutput(
            output_dir=test_dir,
            xy_stations=xy_array,
            idx_stations=idx_array,
            first_date=first_date,
        )
        assert isinstance(test_his_output, OutputProtocol)
        assert isinstance(test_his_output, _CoralOutput)
        assert isinstance(test_his_output, BaseOutput)
        assert isinstance(test_his_output.output_params, CoralOutputParameters)
        assert test_his_output.output_filepath == test_dir / "CoralModel_his.nc"
        assert (test_his_output.xy_stations == xy_array).all()
        assert (test_his_output.idx_stations == idx_array).all()
        assert test_his_output.first_date == first_date
