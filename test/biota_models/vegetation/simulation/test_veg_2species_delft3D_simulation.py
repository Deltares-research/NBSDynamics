from pathlib import Path

from src.biota_models.vegetation.simulation.veg_simulation_2species import _VegetationSimulation_2species
from src.core.simulation.multiplebiota_simulation_protocol import MultipleBiotaSimulationProtocol
from test.utils import TestUtils
from typing import Union
import pytest
from src.biota_models.vegetation.model.veg_constants import VegetationConstants
from src.biota_models.vegetation.simulation.veg_delft3D_simulation_2species import (
    _VegDelft3DSimulation, VegFlowFmSimulation_2species,
)
from src.biota_models.vegetation.model.veg_model import Vegetation
from src.core.simulation.base_simulation import BaseSimulation


class TestVegDelft3DSimulation:
    def test_veg_delft3D_simulation_ctor(self):
        # test_dir = TestUtils.get_local_test_data_dir("delft3d_case")
        test_dir = TestUtils.get_local_test_data_dir("sm_testcase6")
        dll_repo = TestUtils.get_external_repo("DimrDllDependencies")
        kernels_dir = dll_repo / "kernels"

        assert test_dir.is_dir()
        assert kernels_dir.is_dir()

        test_case = test_dir / "input" / "MinFiles"
        species1 = "Salicornia"
        species2 = "Spartina"
        veg_constants1 = VegetationConstants(species=species1)
        veg_constants2 = VegetationConstants(species=species2)

        vegetation_simulation = VegFlowFmSimulation_2species(
            working_dir=test_dir,
            constants=veg_constants1,
            hydrodynamics=dict(
                working_dir=test_dir / "d3d_work",
                d3d_home=kernels_dir,
                dll_path=kernels_dir / "dflowfm_with_shared" / "bin" / "dflowfm",
                definition_file=test_case / "fm" / "test_case6.mdu",
            ),
            output=dict(
                output_dir=test_dir / "output",
                map_output=dict(output_params=dict()),
                his_output=dict(
                    output_params=dict(),
                ),
                species=species1,
            ),
            output2=dict(
                output_dir=test_dir / "output",
                map_output=dict(output_params=dict()),
                his_output=dict(
                    output_params=dict(),
                ),
            ),
            biota=Vegetation(species=species1, constants=veg_constants1),
            biota2=Vegetation(species=species2, constants=veg_constants2),
        )

        assert isinstance(vegetation_simulation, MultipleBiotaSimulationProtocol)
