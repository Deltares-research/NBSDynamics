import pytest

from src.biota_models.vegetation.model.veg_constants import (
    VegetationConstants,
    default_veg_constants_json,
)
from src.core.common.base_constants import BaseConstants


class TestVegetationConstants:
    constant_species_cases = [
        pytest.param("Spartina"),
        pytest.param("Salicornia"),
        pytest.param("Puccinellia"),
    ]

    @pytest.mark.parametrize("species", constant_species_cases)
    def test_init_vegetation_constants(self, species: str):
        test_veg_constants = VegetationConstants(species=species)
        assert test_veg_constants is not None
        assert isinstance(test_veg_constants, BaseConstants)

    @pytest.mark.parametrize(
        "species",
        constant_species_cases,
    )
    def test_set_constants_from_default_json(self, species):
        # Set initial test data
        test_values_dict = {
            "input_file": default_veg_constants_json,
            "species": species,
        }
        # Validate the initial expectations.
        assert len(test_values_dict.items()) == 2

        # Run test
        VegetationConstants.set_constants_from_default_json(test_values_dict)

        # Validate the final expectations.
        assert test_values_dict is not None
        assert len(test_values_dict.items()) > 2
