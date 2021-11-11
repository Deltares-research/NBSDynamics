from datetime import datetime
from test.utils import TestUtils

import numpy as np

from src.core.output.output_model import (
    BaseOutput,
    HisOutput,
    MapOutput,
    ModelParameters,
)
from src.core.output.output_protocol import OutputProtocol


class TestBaseOutput:
    def test_init_baseoutput_model(self):
        test_dir = TestUtils.get_local_test_data_dir("BaseOutput")
        file_name = "justAFile.nc"
        test_baseoutput = BaseOutput(output_dir=test_dir, output_filename=file_name)
        assert not isinstance(test_baseoutput, OutputProtocol)
        assert isinstance(test_baseoutput.output_params, ModelParameters)
        assert test_baseoutput.output_filepath == test_dir / file_name
        assert test_baseoutput.valid_output() is True


class TestModelParameters:
    def test_init_modelattributes(self):
        test_params = ModelParameters()
        assert test_params.lme is True
        assert test_params.fme is True
        assert test_params.tme is True
        assert test_params.pd is True
        assert test_params.ps is True
        assert test_params.calc is True
        assert test_params.md is True
        assert test_params.valid_output() is True


class TestMapOutput:
    def test_init_mapoutput(self):
        test_dir = TestUtils.get_local_test_data_dir("BaseOutput")
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        test_mapoutput = MapOutput(
            output_dir=test_dir, xy_coordinates=xy_array, first_year=2021
        )
        assert isinstance(test_mapoutput, OutputProtocol)
        assert test_mapoutput.output_filepath == test_dir / "CoralModel_map.nc"
        assert (test_mapoutput.xy_coordinates == xy_array).all()
        assert test_mapoutput.first_year == 2021
        assert test_mapoutput.space == len(xy_array)


class TestHisOutput:
    def test_init_mapoutput(self):
        test_dir = TestUtils.get_local_test_data_dir("BaseOutput")
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        idx_array = np.array([[1, 0], [0, 1]], np.float64)
        first_date = datetime.now()
        test_mapoutput = HisOutput(
            output_dir=test_dir,
            xy_stations=xy_array,
            idx_stations=idx_array,
            first_date=first_date,
        )
        assert isinstance(test_mapoutput, OutputProtocol)
        assert test_mapoutput.output_filepath == test_dir / "CoralModel_his.nc"
        assert (test_mapoutput.xy_stations == xy_array).all()
        assert (test_mapoutput.idx_stations == idx_array).all()
        assert test_mapoutput.first_date == first_date
