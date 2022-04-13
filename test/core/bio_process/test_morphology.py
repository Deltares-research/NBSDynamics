from test.core.bio_process.bio_utils import (
    coral_2x2,
    matrix_1x1,
    matrix_2x2,
    valid_coral,
)

import pytest

from src.core.bio_process.morphology import Morphology
from src.coral.model.coral_model import Coral
from src.coral.model.coral_constants import CoralConstants
from src.core.common.space_time import DataReshape


class TestMorphology:
    @pytest.fixture
    def valid_morphology(self, matrix_1x1: DataReshape) -> Morphology:
        assert matrix_1x1.spacetime == (1, 1)
        return Morphology(calc_sum=2.4, light_in=2.4, dt_year=1)

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

    def test_initiation(self, matrix_1x1: DataReshape):
        assert matrix_1x1.spacetime == (1, 1)
        morphology = Morphology(1, 600)
        assert morphology.calc_sum == 1
        assert morphology.I0 == 600
        assert morphology.dt_year == 1
        assert morphology.vol_increase == 0

        assert morphology.rf_optimal is None
        assert morphology.rp_optimal is None
        assert morphology.rs_optimal is None

    @pytest.fixture(autouse=False)
    def mor_legacy(self, matrix_1x1: DataReshape) -> Morphology:
        return Morphology(1, matrix_1x1.variable2matrix(600, "time"))

    def test_calc_sum(self):
        morphology = Morphology([1, 1], 600)
        answer = [1, 1]
        for i, val in enumerate(answer):
            assert morphology.calc_sum[i] == val

    def test_optimal_ratios(
        self, mor_legacy: Morphology, valid_coral: Coral, matrix_1x1: DataReshape
    ):
        valid_coral.light = matrix_1x1.variable2matrix(600, "time")
        valid_coral.ucm = matrix_1x1.variable2array(0.1)

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

    def test_volume_increase(
        self, mor_legacy: Morphology, valid_coral: Coral, matrix_1x1: DataReshape
    ):
        valid_coral.light_bc = matrix_1x1.variable2matrix(0.3, "time")
        mor_legacy.delta_volume(valid_coral)
        assert float(mor_legacy.vol_increase), pytest.approx(8.4375e-6)

    def test_morphology_update(
        self, mor_legacy: Morphology, valid_coral: Coral, matrix_1x1: DataReshape
    ):
        valid_coral.light = matrix_1x1.variable2matrix(600, "time")
        valid_coral.ucm = matrix_1x1.variable2array(0.1)
        valid_coral.light_bc = matrix_1x1.variable2matrix(0.3, "time")
        # morphology.delta_volume(coral)
        for ratio in ("rf", "rp", "rs"):
            setattr(mor_legacy, f"{ratio}_optimal", valid_coral)
        mor_legacy.update(valid_coral)
        assert float(valid_coral.rf), pytest.approx(1.498140551)
        assert float(valid_coral.rp), pytest.approx(0.499964271)
        assert float(valid_coral.rs), pytest.approx(0.666145658)
        assert float(valid_coral.volume), pytest.approx(0.005898924)


class TestMorphology2x2:
    """
    Legacy tests with a DataReshape 2x2 matrix.
    """

    @pytest.fixture(autouse=False)
    def mor_2x2(self, matrix_2x2: DataReshape) -> Morphology:
        assert matrix_2x2.spacetime == (2, 2)
        return Morphology([1, 1], [600, 600])

    def test_initiation(self, mor_2x2: Morphology, matrix_2x2: DataReshape):
        for i in range(matrix_2x2.space):
            assert mor_2x2.calc_sum[i] == 1
            for j in range(matrix_2x2.time):
                assert mor_2x2.I0[i, j] == 600
        assert mor_2x2.dt_year == 1
        assert mor_2x2.vol_increase == 0

        assert mor_2x2.rf_optimal is None
        assert mor_2x2.rp_optimal is None
        assert mor_2x2.rs_optimal is None

    def test_calc_sum_init1(self, matrix_2x2: DataReshape):
        morphology = Morphology([[1, 1], [1, 1]], [600, 600])
        for i in range(matrix_2x2.space):
            assert morphology.calc_sum[i] == 2

    @pytest.mark.skip(reason="Legacy test.")
    def test_optimal_ratios(
        self, mor_2x2: Morphology, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        # TODO: Test failing because the var_optimal setters on morphology
        # TODO: do not seem to be handling the variables as 2x2.
        # TODO: It is not clear if some logic was changed before migration.
        mor_2x2.constants = coral_2x2.constants
        coral_2x2.initiate_coral_morphology()
        coral_2x2.light = matrix_2x2.variable2matrix([600, 600], "time")
        coral_2x2.ucm = matrix_2x2.variable2array([0.1, 0.1])
        ratios = ("rf", "rp", "rs")
        answers = [
            0.2,
            0.475020813,
            0.302412911,
        ]

        for item, ratio in enumerate(ratios):
            setattr(mor_2x2, f"{ratio}_optimal", coral_2x2)
            for i in range(matrix_2x2.space):
                assert float(getattr(mor_2x2, f"{ratio}_optimal")[i]), pytest.approx(
                    answers[item]
                )

    def test_volume_increase(
        self, mor_2x2: Morphology, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        coral_2x2.initiate_coral_morphology()
        coral_2x2.light_bc = DataReshape().variable2matrix(0.3, "time")
        mor_2x2.delta_volume(coral_2x2)
        for i in range(matrix_2x2.space):
            assert float(mor_2x2.vol_increase[i]), pytest.approx(8.4375e-6)

    def test_morphology_update(
        self, mor_2x2: Morphology, coral_2x2: Coral, matrix_2x2: DataReshape
    ):
        coral_2x2.initiate_coral_morphology()
        coral_2x2.light = matrix_2x2.variable2matrix([600, 600], "time")
        coral_2x2.ucm = matrix_2x2.variable2array([0.1, 0.1])
        coral_2x2.light_bc = matrix_2x2.variable2matrix([0.3, 0.3], "time")
        mor_2x2.delta_volume(coral_2x2)
        for ratio in ("rf", "rp", "rs"):
            setattr(mor_2x2, f"{ratio}_optimal", coral_2x2)
        mor_2x2.update(coral_2x2)
        for i in range(matrix_2x2.space):
            assert float(coral_2x2.rf[i]), pytest.approx(1.498140551)
            assert float(coral_2x2.rp[i]), pytest.approx(0.499964271)
            assert float(coral_2x2.rs[i]), pytest.approx(0.666145658)
            assert float(coral_2x2.volume[i]), pytest.approx(0.005898924)
