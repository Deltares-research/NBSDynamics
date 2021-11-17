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

    def test_init_transect_with_args(self):
        # 1. Define test data.
        test_dict = dict(
            working_dir=Path.cwd(),
            definition_file=Path.cwd() / "def_file",
            config_file=Path.cwd() / "conf_file",
        )
        hydromodel = Transect(**test_dict)
        assert hydromodel.working_dir == test_dict["working_dir"]
        assert hydromodel.definition_file == test_dict["definition_file"]
        assert hydromodel.config_file == test_dict["config_file"]

    def test_update_with_stormcat_4_raises_valueerror(self):
        test_trans = Transect()
        with pytest.raises(ValueError) as e_info:
            test_trans.update(coral=None, stormcat=4)
        assert str(e_info.value) == "stormcat = 4, must be either 0,1,2,3"

    def test_update_with_stormcat_4_raises_valueerror(self):
        test_trans = Transect()
        with pytest.raises(ValueError) as e_info:
            test_trans.update(coral=None, stormcat=4)
        assert str(e_info.value) == "stormcat = 4, must be either 0,1,2,3"
