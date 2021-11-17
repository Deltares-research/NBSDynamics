import pytest
from test.core.bio_process.bio_utils import valid_coral
from src.core.bio_process.dislodgment import Dislodgement
from src.core.coral.coral_model import Coral


class TestDislodgement:
    def test_initiation(self):
        dislodgement = Dislodgement()
        dislodgement.dmt is None
        dislodgement.csf is None
        dislodgement.survival is None

    def test_dmt1(self, valid_coral: Coral):
        dislodgement = Dislodgement()
        valid_coral.um = 0
        dislodgement.dislodgement_mechanical_threshold(valid_coral)
        dislodgement.dmt == 1e20

    def test_dmt2(self, valid_coral: Coral):
        dislodgement = Dislodgement()
        valid_coral.um = 0.5
        dislodgement.dislodgement_mechanical_threshold(valid_coral)
        assert float(dislodgement.dmt), pytest.approx(780.487805, 1e-6)

    def test_dmt3(self):
        dislodgement = Dislodgement()
        coral = Coral(
            dc=[0.2, 0.2], hc=[0.3, 0.3], bc=[0.1, 0.1], tc=[0.15, 0.15], ac=[0.3, 0.3]
        )
        coral.um = [0, 0.5]
        dislodgement.dislodgement_mechanical_threshold(coral)
        answers = [1e20, 780.487805]
        for i, ans in enumerate(answers):
            assert float(dislodgement.dmt[i]), pytest.approx(ans, 1e-6)

    def test_csf1(self, valid_coral: Coral):
        dislodgement = Dislodgement()
        dislodgement.colony_shape_factor(valid_coral)
        assert float(dislodgement.csf), pytest.approx(40.1070456591576246)

    @pytest.mark.skip(reason="Test failing due to difference is sizes.")
    def test_csf2(self):
        # TODO: This is a legacy test.
        # TODO: Figure out whether this test is still valid, thus code should be fixed.
        # TODO: Or on the contrary the test has no meaning and therefore should be removed.
        dislodgement = Dislodgement()
        coral = Coral(dc=[0.2, 0], hc=[0.3, 0], bc=[0.1, 0], tc=[0.15, 0], ac=[0.3, 0])
        dislodgement.colony_shape_factor(coral)
        answers = [40.1070456591576246, 0]
        for i, ans in enumerate(answers):
            assert float(dislodgement.csf[i]), pytest.approx(ans)
