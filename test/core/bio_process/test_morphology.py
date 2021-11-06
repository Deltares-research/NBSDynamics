import pytest
from src.core.bio_process.morphology import Morphology
from src.core.coral_model import Coral


class TestMorphology:
    def test_set_rf_optimal_raises_typeerror(self):
        test_morphology = Morphology(None, None, None, None)
        with pytest.raises(TypeError) as e_info:
            test_morphology.rf_optimal = None
        assert (
            str(e_info.value)
            == "The optimal ratios are set using the Coral-object, <class 'NoneType'> is given."
        )

    def test_set_rf_optimal_raises_valueerror(self):
        test_morphology = Morphology(None, None, None, None)
        with pytest.raises(AttributeError) as e_info:
            test_coral = Coral(0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 1)
            delattr(test_coral, "light")
            delattr(test_coral, "ucm")
            test_morphology.rf_optimal = test_coral
        assert str(e_info.value) == (
            "The optimal ratios are determined based on the coral's light and flow conditions; none are provided."
        )
