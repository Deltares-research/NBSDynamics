from pathlib import Path
from test.utils import TestUtils
from typing import Any, Callable, Optional, Union

import pytest

from src.core.biota.coral.coral_model import Coral
from src.core.common.base_constants import BaseConstants
from src.core.common.environment import Environment
from src.core.hydrodynamics.delft3d import DimrModel, FlowFmModel
from src.core.hydrodynamics.factory import HydrodynamicsFactory
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.hydrodynamics.transect import Transect
from src.core.simulation.base_simulation import BaseSimulation, Simulation
from src.core.simulation.simulation_protocol import SimulationProtocol


class TestBaseSimulation:
    @pytest.mark.parametrize(
        "model_type",
        [
            pytest.param(DimrModel, id="DIMR model"),
            pytest.param(FlowFmModel, id="FlowFM model"),
            pytest.param(Transect, id="Transect"),
        ],
    )
    def test_validate_hydrodynamics_present_given_hydromodel(
        self, model_type: HydrodynamicProtocol
    ):
        model_in = model_type()
        model_out = BaseSimulation.validate_hydrodynamics_present(model_in, dict())
        assert isinstance(model_out, HydrodynamicProtocol)
        assert model_in == model_out


class TestVanillaSimulation:
    @pytest.mark.parametrize("mode", HydrodynamicsFactory.supported_modes)
    def test_init_vanilla_simulation(self, mode: HydrodynamicProtocol):
        test_sim = Simulation(mode=mode.__name__)
        assert isinstance(test_sim.hydrodynamics, mode)
        # assert isinstance(test_sim, SimulationProtocol)
        assert isinstance(test_sim.environment, Environment)
        assert test_sim.constants is None
        assert isinstance(test_sim.working_dir, Path)
        # Verify abstract methods raise nothing.
        test_sim.configure_hydrodynamics()
        test_sim.configure_output()
