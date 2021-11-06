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
