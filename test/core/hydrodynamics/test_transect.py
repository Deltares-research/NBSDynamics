from pathlib import Path

import pytest

from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
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
        assert isinstance(test_transect, HydrodynamicProtocol)
        assert test_transect.time_step is None
        assert test_transect.settings == expected_settings
        assert test_transect.working_dir is None
        assert test_transect.definition_file is None
        assert test_transect.config_file is None
        assert test_transect.x_coordinates is None
        assert test_transect.y_coordinates is None
        assert test_transect.xy_coordinates is None
        assert test_transect.space is None
        assert test_transect.water_depth is None
        assert repr(test_transect) == "Transect()"

    def test_set_workdir_as_str_returns_path(self):
        test_trans = Transect()
        test_trans.working_dir = "thisPath"
        assert isinstance(test_trans.working_dir, Path)

    def test_set_definition_file_relative_to_work_dir(self):
        test_trans = Transect()
        test_trans.working_dir = "thisPath"
        test_trans.definition_file = "anMdu"
        assert test_trans.definition_file == Path("thisPath") / "anMdu"

    def test_set_config_relative_to_work_dir(self):
        test_trans = Transect()
        test_trans.working_dir = "thisPath"
        test_trans.config_file = "config"
        assert test_trans.config_file == Path("thisPath") / "config"

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
