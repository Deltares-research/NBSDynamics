from pathlib import Path

import pytest

from src.core.hydrodynamics.delft3d import Delft3D


class TestDelft3d:
    def test_init_delft3d(self):
        test_delft3d = Delft3D()
        assert test_delft3d.time_step == None
        assert test_delft3d.d3d_home == None
        assert test_delft3d.working_dir == None
        assert test_delft3d.mdu == None
        assert test_delft3d.config == None

    def test_set_d3d_home_sets_other_paths(self):
        test_delft3d = Delft3D()
        ref_path = Path()
        test_delft3d.d3d_home = ref_path
        assert test_delft3d.dflow_dir == ref_path / "dflowfm" / "bin" / "dflowfm"
        assert test_delft3d.dimr_dir == ref_path / "dimr" / "bin" / "dimr_dll"

    def test_model_fm_no_model_raises_valueeror(self):
        test_delft3d = Delft3D()
        with pytest.raises(ValueError) as e_info:
            test_delft3d.model_fm
        assert str(e_info.value) == "Model FM has not been defined."

    def test_model_dimr_no_model_raises_valueeror(self):
        test_delft3d = Delft3D()
        with pytest.raises(ValueError) as e_info:
            test_delft3d.model_dimr
        assert str(e_info.value) == "Model dimr has not been defined."

    def test_settings_returns_expected_values(self):
        expected_text = (
            "Coupling with Delft3D model (incl. DFlow-module) with the following settings:"
            "\n\tDelft3D home dir.  : None"
            "\n\tDFlow file         : None"
        )
        test_delft3d = Delft3D()
        assert test_delft3d.settings == expected_text

    def test_settings_returns_expected_values_config_true(self):
        expected_text = (
            "Coupling with Delft3D model (incl. DFlow- and DWaves-modules) with the following settings:"
            "\n\tDelft3D home dir.  : None"
            "\n\tDFlow file         : None"
            "\n\tConfiguration file : aPath"
        )
        test_delft3d = Delft3D()
        test_delft3d.working_dir = Path()
        test_delft3d.config = "aPath"
        assert test_delft3d.settings == expected_text
