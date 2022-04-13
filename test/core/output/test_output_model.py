from test.utils import TestUtils

from src.core.output.base_output_model import BaseOutput, BaseOutputParameters
from src.core.output.output_protocol import OutputProtocol


class TestBaseOutput:
    def test_init_baseoutput_model(self):
        test_dir = TestUtils.get_local_test_data_dir("BaseOutput")
        file_name = "justAFile.nc"
        test_baseoutput = BaseOutput(output_dir=test_dir, output_filename=file_name)
        assert not isinstance(test_baseoutput, OutputProtocol)
        assert isinstance(test_baseoutput.output_params, BaseOutputParameters)
        assert test_baseoutput.output_filepath == test_dir / file_name
        assert test_baseoutput.valid_output() is True
