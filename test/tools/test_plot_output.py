import pytest
from src.tools.plot_output import plot_output


class TestPlotOutput:
    def test_plot_output_raises_error(self):
        with pytest.raises(ValueError) as e_err:
            plot_output(None)
        assert str(e_err.value) == "No output model provided."
