from pathlib import Path
from test.utils import TestUtils
from typing import Union

import pytest

from src.biota_models.vegetation.model.veg_constants import VegetationConstants
from src.biota_models.vegetation.model.veg_model import Vegetation
from src.biota_models.vegetation.output.veg_output_wrapper import VegOutputWrapper
from src.biota_models.vegetation.simulation.veg_delft3D_simulation_2species import (
    VegFlowFmSimulation_2species,
    _VegDelft3DSimulation,
)
from src.biota_models.vegetation.simulation.veg_simulation_2species import (
    VegetationBiotaWrapper,
    _VegetationSimulation_2species,
)
from src.core.biota.biota_model import Biota
from src.core.simulation.base_simulation import BaseSimulation
from src.core.simulation.biota_wrapper import BiotaWrapper
from src.core.simulation.multiplebiota_base_simulation import (
    MultipleBiotaBaseSimulation,
)
from src.core.simulation.multiplebiota_simulation_protocol import (
    MultipleBiotaSimulationProtocol,
)


class TestVegDelft3DSimulation:
    def test_veg_delft3D_simulation_ctor(self):
        test_dir = TestUtils.get_local_test_data_dir("sm_testcase6")
        # dll_repo = TestUtils.get_external_repo("DimrDllDependencies")
        # kernels_dir = dll_repo / "kernels"

        assert test_dir.is_dir()
        # assert kernels_dir.is_dir()

        test_case = test_dir / "input" / "MinFiles"
        salicornia_species = "Salicornia"
        spartina_species = "Spartina"
        main_veg_constants = VegetationConstants(species=salicornia_species)

        vegetation_simulation = VegFlowFmSimulation_2species(
            working_dir=test_dir,
            constants=main_veg_constants,
            hydrodynamics=dict(
                working_dir=test_dir / "d3d_work",
                # For an integration test is not required to provide local non-versioned data.
                # d3d_home=kernels_dir,
                # dll_path=kernels_dir / "dflowfm_with_shared" / "bin" / "dflowfm",
                definition_file=test_case / "fm" / "test_case6.mdu",
            ),
            biota_wrapper_list=[
                VegetationBiotaWrapper(
                    biota=Vegetation(
                        species=salicornia_species, constants=main_veg_constants
                    ),
                    output=dict(
                        output_dir=test_dir / "output",
                        map_output=dict(output_params=dict()),
                        his_output=dict(
                            output_params=dict(),
                        ),
                        species=salicornia_species,
                    ),
                ),
                VegetationBiotaWrapper(
                    output=dict(
                        output_dir=test_dir / "output",
                        map_output=dict(output_params=dict()),
                        his_output=dict(
                            output_params=dict(),
                        ),
                    ),
                    biota=Vegetation(
                        species=spartina_species,
                        constants=VegetationConstants(species=spartina_species),
                    ),
                ),
            ],
        )

        assert isinstance(vegetation_simulation, MultipleBiotaSimulationProtocol)
        assert isinstance(vegetation_simulation, MultipleBiotaBaseSimulation)
        assert isinstance(vegetation_simulation, _VegDelft3DSimulation)
        assert len(vegetation_simulation.biota_wrapper_list) == 2
        for biota_wrapper in vegetation_simulation.biota_wrapper_list:
            assert isinstance(biota_wrapper, BiotaWrapper)
            assert isinstance(biota_wrapper, VegetationBiotaWrapper)
            assert isinstance(biota_wrapper.biota, Vegetation)
            assert isinstance(biota_wrapper.biota, Biota)
            assert isinstance(biota_wrapper.output, VegOutputWrapper)
