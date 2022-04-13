from pathlib import Path
from test.utils import TestUtils
from typing import Union

import pytest

from src.core.common.coral_constants import CoralConstants
from src.core.simulation.base_simulation import BaseSimulation
from src.core.simulation.coral_transect_simulation import CoralTransectSimulation


class TestCoralTransectSimulation:
    def test_coral_transect_simulation_ctor(self):
        test_sim = CoralTransectSimulation()
        assert issubclass(type(test_sim), BaseSimulation)
        assert test_sim.mode == "Transect"
