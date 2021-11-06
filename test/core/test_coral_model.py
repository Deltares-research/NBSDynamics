import pytest
from src.core.coral_model import Coral
import numpy as np


class TestCoralModel:
    @pytest.fixture(autouse=True)
    def coral_model_test(self) -> Coral:
        input_dict = dict(
            constants=None,
            dc=0.2,
            hc=0.1,
            bc=0.2,
            tc=0.1,
            ac=0.2,
            species_constant=1,
        )
        return Coral(**input_dict)

    def test_init_coral_model(self, coral_model_test: Coral):
        assert repr(coral_model_test) == "Morphology([0.2], [0.1], [0.2], [0.2], [0.2])"
        assert str(coral_model_test) == (
            "Coral morphology with: dc = [0.2] m; hc = [0.1] ;bc = [0.2] m; tc = [0.1] m; ac = [0.2] m"
        )

    def test_set_cover(self, coral_model_test: Coral):
        coral_model_test.volume = np.array([4.2])
        coral_model_test.cover = 2
        assert coral_model_test.cover == 2

    def test_set_cover_odd_shape_raises_error(self, coral_model_test: Coral):
        with pytest.raises(ValueError) as e_info:
            coral_model_test.cover = np.array([4, 2])
        assert str(e_info.value) == "Shapes do not match: (1,) =/= (2,)"

    def test_initiate_spatial_morphology_with_invalid_cover_raises(
        self, coral_model_test: Coral
    ):
        with pytest.raises(ValueError) as e_info:
            coral_model_test.initiate_spatial_morphology(np.array([4, 2]))
        assert (
            str(e_info.value)
            == "Spatial dimension of cover does not match: (2,) =/= 1."
        )

    def test_initiate_spatial_morphology_with_cover(self, coral_model_test: Coral):
        coral_model_test.initiate_spatial_morphology(np.array([4]))
