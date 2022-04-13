from datetime import datetime
from test.utils import TestUtils

import numpy as np

from src.core.output.base_output_model import BaseOutput
from src.core.output.output_protocol import OutputProtocol
from src.vegetation.output.veg_output_model import (
    VegetationHisOutput,
    VegetationMapOutput,
    VegetationOutputParameters,
    _VegetationOutput,
)


class TestVegetationOutputParameters:
    def test_init_modelattributes(self):
        test_params = VegetationOutputParameters()
        assert test_params.hydro_mor is True
        assert test_params.veg_characteristics is True
        assert test_params.valid_output()


class TestCoralMapOutput:
    def test_init_mapoutput(self):
        test_dir = TestUtils.get_local_test_data_dir("BaseOutput")
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        test_map_output = VegetationMapOutput(
            output_dir=test_dir, xy_coordinates=xy_array, first_year=2021
        )
        assert isinstance(test_map_output, OutputProtocol)
        assert isinstance(test_map_output, _VegetationOutput)
        assert isinstance(test_map_output, BaseOutput)
        assert isinstance(test_map_output.output_params, VegetationOutputParameters)

        assert test_map_output.output_filepath == test_dir / "VegModel_map.nc"
        assert (test_map_output.xy_coordinates == xy_array).all()
        assert test_map_output.first_year == 2021
        assert test_map_output.space == len(xy_array)


class TestVegetationHisOutput:
    def test_init_mapoutput(self):
        test_dir = TestUtils.get_local_test_data_dir("BaseOutput")
        xy_array = np.array([[0, 1], [1, 0]], np.float64)
        idx_array = np.array([[1, 0], [0, 1]], np.float64)
        first_date = datetime.now()
        test_his_output = VegetationHisOutput(
            output_dir=test_dir,
            xy_stations=xy_array,
            idx_stations=idx_array,
            first_date=first_date,
        )
        assert isinstance(test_his_output, OutputProtocol)
        assert isinstance(test_his_output, _VegetationOutput)
        assert isinstance(test_his_output, BaseOutput)
        assert isinstance(test_his_output.output_params, VegetationOutputParameters)
        assert test_his_output.output_filepath == test_dir / "VegModel_his.nc"
        assert (test_his_output.xy_stations == xy_array).all()
        assert (test_his_output.idx_stations == idx_array).all()
        assert test_his_output.first_date == first_date
