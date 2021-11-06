from pathlib import Path
import pytest

from src.core.hydrodynamics.transect import Transect


class TestTransect:
    def test_init_transect(self):
        test_transect = Transect()
        expected_settings = (
            "1D schematic cross-shore transect with forced hydrodynamics: "
            "\n\tTransect model working dir.  : None"
            "\n\tTransect config file  : None"
            "\n\tTransect forcings file : None"
        )
        assert test_transect.time_step is None
        assert test_transect.settings == expected_settings
        assert test_transect.working_dir is None
        assert test_transect.mdu is None
        assert test_transect.config is None
        assert repr(test_transect) == "Transect()"

    def test_set_workdir_as_str_returns_path(self):
        test_Transect = Transect()
        test_Transect.working_dir = "thisPath"
        assert isinstance(test_Transect.working_dir, Path)

    def test_set_mdu_relative_to_work_dir(self):
        test_Transect = Transect()
        test_Transect.working_dir = "thisPath"
        test_Transect.mdu = "anMdu"
        assert test_Transect.mdu == Path("thisPath") / "anMdu"

    def test_set_config_relative_to_work_dir(self):
        test_Transect = Transect()
        test_Transect.working_dir = "thisPath"
        test_Transect.config = "config"
        assert test_Transect.config == Path("thisPath") / "config"

    def test_update_with_stormcat_4_raises_valueerror(self):
        test_trans = Transect()
        with pytest.raises(ValueError) as e_info:
            test_trans.update(coral=None, stormcat=4)
        assert str(e_info.value) == "stormcat = 4, must be either 0,1,2,3"

    def test_update_orbital_with_stormcat_4_raises_valueerror(self):
        test_trans = Transect()
        with pytest.raises(ValueError) as e_info:
            test_trans.update_orbital(stormcat=4)
        assert str(e_info.value) == "stormcat = 4, must be either 0,1,2,3"
