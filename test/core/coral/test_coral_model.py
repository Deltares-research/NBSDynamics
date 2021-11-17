import numpy as np
import pytest

from src.core.constants import Constants
from src.core.coral.coral_model import Coral
from src.core.coral.coral_protocol import CoralProtocol


class TestCoralModel:
    @pytest.fixture(autouse=False)
    def coral_model_test(self) -> CoralProtocol:
        input_dict = dict(
            constants=Constants(),
            dc=0.2,
            hc=0.1,
            bc=0.2,
            tc=0.1,
            ac=0.2,
            Csp=1,
        )
        return Coral(**input_dict)

    def test_init_coral_model(self, coral_model_test: Coral):
        assert isinstance(coral_model_test, CoralProtocol)
        assert repr(coral_model_test) == "Morphology([0.2], [0.1], [0.2], [0.2], [0.2])"
        assert str(coral_model_test) == (
            "Coral morphology with: dc = [0.2] m; hc = [0.1] ;bc = [0.2] m; tc = [0.1] m; ac = [0.2] m"
        )
        assert coral_model_test.dc[0] == 0.2
        assert coral_model_test.hc[0] == 0.1
        assert coral_model_test.bc[0] == 0.2
        assert coral_model_test.tc[0] == 0.1
        assert coral_model_test.ac[0] == 0.2
        assert coral_model_test.Csp == 1
        eps_err = 0.0000001
        assert float(coral_model_test.dc_rep), pytest.approx(0.2, eps_err)
        assert float(coral_model_test.rf), pytest.approx(0.5, eps_err)
        assert float(coral_model_test.rp), pytest.approx(1, eps_err)
        assert float(coral_model_test.rs), pytest.approx(1, eps_err)
        assert coral_model_test.volume, pytest.approx(0.00314159, eps_err)
        assert coral_model_test.as_vegetation_density, pytest.approx(10, eps_err)
        assert coral_model_test.cover, pytest.approx(1, eps_err)

    def test_set_cover(self, coral_model_test: Coral):
        coral_model_test.update_coral_volume(np.array([4.2]))
        coral_model_test.update_cover(2)
        assert coral_model_test.cover == 2

    def test_set_cover_odd_shape_raises_error(self, coral_model_test: Coral):
        with pytest.raises(ValueError) as e_info:
            coral_model_test.update_cover(np.array([4, 2]))
        assert str(e_info.value) == "Shapes do not match: (1,) =/= (2,)"

    def test_initiate_coral_morphology_with_invalid_cover_raises(
        self, coral_model_test: Coral
    ):
        with pytest.raises(ValueError) as e_info:
            coral_model_test.initiate_coral_morphology(np.array([4, 2]))
        assert (
            str(e_info.value)
            == "Spatial dimension of cover does not match: (2,) =/= 1."
        )

    def test_initiate_coral_morphology_with_cover(self, coral_model_test: Coral):
        coral_model_test.initiate_coral_morphology(np.array([4]))
