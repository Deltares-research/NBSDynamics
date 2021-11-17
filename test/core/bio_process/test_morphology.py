from test.core.bio_process.bio_utils import valid_coral

import pytest

from src.core.bio_process.morphology import Morphology
from src.core.coral.coral_model import Coral
from src.core.utils import DataReshape
from src.core.constants import Constants


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

    def test_initiation(self):
        morphology = Morphology(Constants(), 1, 600, DataReshape())
        assert morphology.calc_sum == 1
        assert morphology.I0 == 600
        assert morphology.dt_year == 1
        assert morphology.vol_increase == 0

        assert morphology.rf_optimal is None
        assert morphology.rp_optimal is None
        assert morphology.rs_optimal is None

    @pytest.fixture(autouse=True)
    def mor_legacy(self) -> Morphology:
        return Morphology(
            Constants(), 1, DataReshape().variable2matrix(600, "time"), DataReshape()
        )

    def test_calc_sum(self):
        morphology = Morphology(Constants(), [1, 1], 600, DataReshape())
        answer = [1, 1]
        for i, val in enumerate(answer):
            assert morphology.calc_sum[i] == val

    def test_optimal_ratios(self, mor_legacy: Morphology, valid_coral: Coral):
        valid_coral.light = DataReshape().variable2matrix(600, "time")
        valid_coral.ucm = DataReshape().variable2array(0.1)

        ratios = ("rf", "rp", "rs")
        answers = [
            0.2,
            0.475020813,
            0.302412911,
        ]

        for i, ratio in enumerate(ratios):
            setattr(mor_legacy, f"{ratio}_optimal", valid_coral)
            assert float(getattr(mor_legacy, f"{ratio}_optimal")), pytest.approx(
                answers[i]
            )

    def test_volume_increase(self, mor_legacy: Morphology, valid_coral: Coral):
        valid_coral.light_bc = DataReshape().variable2matrix(0.3, "time")
        mor_legacy.delta_volume(valid_coral)
        assert float(mor_legacy.vol_increase), pytest.approx(8.4375e-6)

    def test_morphology_update(self, mor_legacy: Morphology, valid_coral: Coral):
        valid_coral.light = DataReshape().variable2matrix(600, "time")
        valid_coral.ucm = DataReshape().variable2array(0.1)
        valid_coral.light_bc = DataReshape().variable2matrix(0.3, "time")
        # morphology.delta_volume(coral)
        for ratio in ("rf", "rp", "rs"):
            setattr(mor_legacy, f"{ratio}_optimal", valid_coral)
        mor_legacy.update(valid_coral)
        assert float(valid_coral.rf), pytest.approx(1.498140551)
        assert float(valid_coral.rp), pytest.approx(0.499964271)
        assert float(valid_coral.rs), pytest.approx(0.666145658)
        assert float(valid_coral.volume), pytest.approx(0.005898924)
