from pathlib import Path

from src.core.hydrodynamics.transect import Transect
from src.core.simulation.base_simulation import BaseSimulation
from src.core.simulation.coral_transect_simulation import CoralTransectSimulation


class TestCoralTransectSimulation:
    def test_coral_transect_simulation_ctor(self):
        test_sim = CoralTransectSimulation()
        assert issubclass(type(test_sim), BaseSimulation)
        assert test_sim.mode == "Transect"

    def test_set_simulation_hydrodynamics(self):
        # 1. Define test data.
        test_dict = dict(
            working_dir=Path.cwd(),
            definition_file=Path.cwd() / "def_file",
            config_file=Path.cwd() / "conf_file",
        )
        hydromodel = Transect()
        assert hydromodel.working_dir is None
        assert hydromodel.definition_file is None
        assert hydromodel.config_file is None

        # 2. Run test
        CoralTransectSimulation.set_simulation_hydrodynamics(hydromodel, test_dict)

        # 3. Verify final expectations
        assert hydromodel.working_dir == test_dict["working_dir"]
        assert hydromodel.definition_file == test_dict["definition_file"]
        assert hydromodel.config_file == test_dict["config_file"]

    def test_set_simulation_hydrodynamics_no_entries_raises_nothing(self):
        hydromodel = Transect()
        CoralTransectSimulation.set_simulation_hydrodynamics(
            hydromodel, dict(working_dir=Path.cwd())
        )
        assert hydromodel.working_dir == Path.cwd()
        assert hydromodel.definition_file is None
        assert hydromodel.config_file is None
