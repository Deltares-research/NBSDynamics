from pathlib import Path
from typing import List

import pytest

from src.core.hydrodynamics.delft3d import Delft3D, DimrModel, FlowFmModel
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.simulation.base_simulation import _Simulation
from src.core.simulation.coral_delft3d_simulation import (
    CoralDimrSimulation,
    CoralFlowFmSimulation,
    _CoralDelft3DSimulation,
)

hydrodynamic_model_cases: List[pytest.param] = [
    pytest.param(DimrModel, id="Dimr Model"),
    pytest.param(FlowFmModel, id="FlowFM Mode"),
]


class TestCoralDelft3dSimulation:
    @pytest.mark.parametrize(
        "coral_mode, expected_mode",
        [
            pytest.param(CoralDimrSimulation, "DimrModel", id="Coral DIMR"),
            pytest.param(CoralFlowFmSimulation, "FlowFMModel", id="Coral FlowFM"),
        ],
    )
    def test_delft3d_ctor(
        self, coral_mode: _CoralDelft3DSimulation, expected_mode: str
    ):
        test_coral = coral_mode()
        assert issubclass(coral_mode, _CoralDelft3DSimulation)
        assert issubclass(coral_mode, _Simulation)
        assert test_coral.mode == expected_mode

    @pytest.mark.parametrize("hydrodynamic_model", hydrodynamic_model_cases)
    def test_set_simulation_hydrodynamics(
        self, hydrodynamic_model: HydrodynamicProtocol
    ):
        # 1. Define test data.
        test_dict = dict(
            working_dir=Path.cwd(),
            definition_file=Path.cwd() / "def_file",
            config_file=Path.cwd() / "conf_file",
            d3d_home=Path.cwd() / "d3d_home",
        )
        hydromodel: Delft3D = hydrodynamic_model()
        assert hydromodel.working_dir is None
        assert hydromodel.definition_file is None
        assert hydromodel.config_file is None
        assert hydromodel.d3d_home is None

        # 2. Run test
        _CoralDelft3DSimulation.set_simulation_hydrodynamics(hydromodel, test_dict)

        # 3. Verify final expectations
        assert hydromodel.working_dir == test_dict["working_dir"]
        assert hydromodel.definition_file == test_dict["definition_file"]
        assert hydromodel.config_file == test_dict["config_file"]
        assert hydromodel.d3d_home == test_dict["d3d_home"]

    @pytest.mark.parametrize("hydrodynamic_model", hydrodynamic_model_cases)
    def test_set_simulation_hydrodynamics_no_entries_raises_nothing(
        self, hydrodynamic_model: HydrodynamicProtocol
    ):
        hydromodel: Delft3D = hydrodynamic_model()
        _CoralDelft3DSimulation.set_simulation_hydrodynamics(
            hydromodel, dict(working_dir=Path.cwd())
        )
        assert hydromodel.working_dir == Path.cwd()
        assert hydromodel.definition_file is None
        assert hydromodel.config_file is None
        assert hydromodel.d3d_home is None
