from test.core.bio_process.bio_utils import valid_coral

import pytest

from src.core.bio_process.morphology import Morphology
from src.core.coral.coral_model import Coral
from src.core.utils import DataReshape


class TestMorphology:
    @pytest.fixture
    def valid_morphology(self) -> Morphology:
        return Morphology(2.4, 2.4, 2.4, DataReshape(), 1)

    def test_set_rf_optimal_raises_typeerror(self, valid_morphology: Morphology):
        with pytest.raises(TypeError) as e_info:
            valid_morphology.rf_optimal = None
        assert (
            str(e_info.value)
            == "The optimal ratios are set using the Coral-object, <class 'NoneType'> is given."
        )

    def test_set_rf_optimal_raises_valueerror(
        self, valid_morphology: Morphology, valid_coral: Coral
    ):
        with pytest.raises(AttributeError) as e_info:
            delattr(valid_coral, "light")
            delattr(valid_coral, "ucm")
            valid_morphology.rf_optimal = valid_coral
        assert str(e_info.value) == (
            "The optimal ratios are determined based on the coral's light and flow conditions; none are provided."
        )

    def test_ratio_update_raises_error(
        self, valid_morphology: Morphology, valid_coral: Coral
    ):
        ratio_value = "notARatio"

        with pytest.raises(ValueError) as e_info:
            valid_morphology.ratio_update(valid_coral, ratio_value)
        assert str(e_info.value) == "notARatio not in ('rf', 'rp', 'rs')."
