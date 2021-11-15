from pathlib import Path
from typing import List

import pytest

from src.core.hydrodynamics.delft3d import Delft3D, DimrModel, FlowFmModel
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol

delft3d_type_cases: List[pytest.param] = [
    pytest.param(DimrModel, id="Dimr Model"),
    pytest.param(FlowFmModel, id="FlowFM Model"),
]


class TestDelft3d:
    @pytest.mark.parametrize(
        "model_type",
        delft3d_type_cases,
    )
    def test_init_delft3d(self, model_type: HydrodynamicProtocol):
        test_delft3d: Delft3D = model_type()
        assert isinstance(test_delft3d, HydrodynamicProtocol)
        assert test_delft3d.model_wrapper is None
        assert test_delft3d.time_step is None
        assert test_delft3d.d3d_home is None
        assert test_delft3d.working_dir is None
        assert test_delft3d.definition_file is None
        assert test_delft3d.config_file is None
        assert test_delft3d.x_coordinates is None
        assert test_delft3d.y_coordinates is None
        assert test_delft3d.xy_coordinates is None
        assert test_delft3d.water_depth is None
        assert test_delft3d.space is None
        assert repr(test_delft3d) == "Delft3D()"

    @pytest.mark.parametrize(
        "model_type, expected_path",
        [
            pytest.param(DimrModel, Path("dimr") / "bin" / "dimr_dll", id="Dimr Model"),
            pytest.param(
                FlowFmModel, Path("dflowfm") / "bin" / "dflowfm", id="FlowFM Model"
            ),
        ],
    )
    def test_set_d3d_home_sets_other_paths(
        self, model_type: HydrodynamicProtocol, expected_path: Path
    ):
        test_delft3d: Delft3D = model_type()
        ref_path = Path()
        test_delft3d.d3d_home = ref_path
        assert test_delft3d.dll_dir == ref_path / expected_path

    def test_flowfm_settings_returns_expected_values(self):
        expected_text = (
            "Coupling with Delft3D model (incl. DFlow-module) with the following settings:"
            "\n\tDelft3D home dir.  : None"
            "\n\tDFlow file         : None"
        )
        test_delft3d = FlowFmModel()
        assert test_delft3d.settings == expected_text

    def test_dimr_settings_returns_expected_values_config_true(self):
        expected_text = (
            "Coupling with Delft3D model (incl. DFlow- and DWaves-modules) with the following settings:"
            "\n\tDelft3D home dir.  : None"
            "\n\tDFlow file         : None"
            "\n\tConfiguration file : aPath"
        )
        test_delft3d = DimrModel()
        test_delft3d.working_dir = Path()
        test_delft3d.config_file = "aPath"
        assert test_delft3d.settings == expected_text

    @pytest.mark.parametrize(
        "model_type",
        delft3d_type_cases,
    )
    def test_set_definition_file_and_config_fige_not_relative_to_work_dir(
        self, model_type: HydrodynamicProtocol
    ):
        test_delft3d: Delft3D = model_type()
        test_delft3d.working_dir = "thisPath"
        test_delft3d.definition_file = "anMdu"
        test_delft3d.config_file = "aConfigFile"
        assert test_delft3d.definition_file != Path("thisPath") / "anMdu"
        assert test_delft3d.config_file != Path("thisPath") / "aConfigFile"
