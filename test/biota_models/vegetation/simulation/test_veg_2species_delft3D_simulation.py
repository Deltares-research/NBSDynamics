from pathlib import Path
from test.utils import TestUtils
from typing import Union

import pytest
from numpy import isin

from src.biota_models.vegetation.model.veg_constants import VegetationConstants
from src.biota_models.vegetation.model.veg_model import Vegetation
from src.biota_models.vegetation.output.veg_output_wrapper import VegOutputWrapper
from src.biota_models.vegetation.simulation.veg_delft3D_simulation_2species import (
    VegFlowFmSimulation_2species,
    _VegDelft3DSimulation,
)
from src.biota_models.vegetation.simulation.veg_simulation_2species import (
    VegetationBiotaWrapper,
)
from src.core.biota.biota_model import Biota
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

    def test_validate_each_biota_wrapper(self):
        validated_wrapper = VegFlowFmSimulation_2species.validate_each_biota_wrapper(
            VegetationBiotaWrapper(), None
        )
        assert isinstance(validated_wrapper, VegetationBiotaWrapper)
        assert not validated_wrapper.biota
        assert not validated_wrapper.output

    def test_validate_each_biota_wrapper_sets_constants_when_not_present(self):
        sim_constants = VegetationConstants(species="Spartina")
        validated_wrapper = VegFlowFmSimulation_2species.validate_each_biota_wrapper(
            dict(biota=dict(species="Spartina")), dict(constants=sim_constants)
        )
        assert isinstance(validated_wrapper, VegetationBiotaWrapper)
        assert isinstance(validated_wrapper.biota, Vegetation)
        assert validated_wrapper.biota.constants == sim_constants

    def test_validate_each_biota_wrapper_returns_biotawrapper(self):
        sim_constants = VegetationConstants(species="Spartina")
        validated_wrapper = VegFlowFmSimulation_2species.validate_each_biota_wrapper(
            dict(biota=dict(species="Spartina", constants=sim_constants)), None
        )
        assert isinstance(validated_wrapper, VegetationBiotaWrapper)
        assert isinstance(validated_wrapper.biota, Vegetation)
        assert validated_wrapper.biota.constants == sim_constants


class TestVegetationBiotaWrapper:
    def test_vegetationbiotawrapper_empty_ctor(self):
        biota_wrapper = VegetationBiotaWrapper()
        assert isinstance(biota_wrapper, BiotaWrapper)
        assert not biota_wrapper.biota
        assert not biota_wrapper.output

    @pytest.mark.parametrize(
        "veg_species", [pytest.param("Salicornia"), pytest.param("Spartina")]
    )
    def test_vegetationbiotawrapper_valid_ctor(self, veg_species: str):
        biota_wrapper = VegetationBiotaWrapper(
            biota=dict(species=veg_species, constants=dict(species=veg_species)),
            output=VegOutputWrapper(),
        )
        assert isinstance(biota_wrapper, BiotaWrapper)
        assert isinstance(biota_wrapper.biota, Vegetation)
        assert isinstance(biota_wrapper.biota, Biota)
        assert isinstance(biota_wrapper.output, VegOutputWrapper)

    def test_validate_vegetation_given_biota(self):
        input_biota = Vegetation(
            species="Spartina",
            constants=VegetationConstants(species="Spartina"),
        )
        assert VegetationBiotaWrapper.validate_vegetation(
            field_value=input_biota, values=None
        )

    def test_validate_vegetation_given_valid_dict(self):
        gen_vegetation = VegetationBiotaWrapper.validate_vegetation(
            field_value=dict(species="Spartina", constants=dict(species="Spartina")),
            values=None,
        )
        assert isinstance(gen_vegetation, Vegetation)
        assert isinstance(gen_vegetation.constants, VegetationConstants)

    def test_validate_vegetation_given_invalid_dict_raises(self):
        with pytest.raises(ValueError) as err_info:
            VegetationBiotaWrapper.validate_vegetation(
                field_value=dict(), values=dict()
            )
        assert (
            str(err_info.value)
            == "Constants should be provided to initialize a Vegetation Model."
        )
