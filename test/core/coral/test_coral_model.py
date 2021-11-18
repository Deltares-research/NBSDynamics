from typing import Tuple

import numpy as np
import pytest

from src.core.common.constants import _Constants as Constants
from src.core.common.singletons import RESHAPE
from src.core.common.space_time import DataReshape
from src.core.coral.coral_model import Coral
from src.core.coral.coral_protocol import CoralProtocol


@pytest.fixture(autouse=False)
def coral_1x1() -> Coral:
    RESHAPE().spacetime = (1, 1)
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


@pytest.fixture(autouse=False)
def coral_2x2() -> Coral:
    RESHAPE().spacetime = (2, 2)
    input_dict = dict(
        RESHAPE=DataReshape((2, 2)),
        constants=Constants(),
        dc=[0.2, 0.2],
        hc=[0.3, 0.3],
        bc=[0.1, 0.1],
        tc=[0.15, 0.15],
        ac=[0.3, 0.3],
        Csp=1,
    )
    return Coral(**input_dict)


class TestCoral1x1:
    @staticmethod
    def _get_coral_model(coral_values: dict, reshape_value: Tuple[int, int]) -> Coral:
        return Coral(**coral_values)

    def test_init_coral_model(self, coral_1x1: Coral):
        assert isinstance(coral_1x1, CoralProtocol)
        assert repr(coral_1x1) == "Morphology([0.2], [0.1], [0.2], [0.2], [0.2])"
        assert str(coral_1x1) == (
            "Coral morphology with: dc = [0.2] m; hc = [0.1] ;bc = [0.2] m; tc = [0.1] m; ac = [0.2] m"
        )

        assert coral_1x1.dc[0] == 0.2
        assert coral_1x1.hc[0] == 0.1
        assert coral_1x1.bc[0] == 0.2
        assert coral_1x1.tc[0] == 0.1
        assert coral_1x1.ac[0] == 0.2
        assert coral_1x1.Csp == 1
        eps_err = 0.0000001
        assert float(coral_1x1.dc_rep), pytest.approx(0.2, eps_err)
        assert float(coral_1x1.rf), pytest.approx(0.5, eps_err)
        assert float(coral_1x1.rp), pytest.approx(1, eps_err)
        assert float(coral_1x1.rs), pytest.approx(1, eps_err)
        assert coral_1x1.volume, pytest.approx(0.00314159, eps_err)
        assert coral_1x1.as_vegetation_density, pytest.approx(10, eps_err)
        assert coral_1x1.cover, pytest.approx(1, eps_err)

    def test_set_cover(self, coral_1x1: Coral):
        coral_1x1.update_coral_volume(np.array([4.2]))
        coral_1x1.update_cover(2)
        assert coral_1x1.cover == 2

    def test_set_cover_odd_shape_raises_error(self, coral_1x1: Coral):
        with pytest.raises(ValueError) as e_info:
            coral_1x1.update_cover(np.array([4, 2]))
        assert str(e_info.value) == "Shapes do not match: (1,) =/= (2,)"

    def test_initiate_coral_morphology_with_invalid_cover_raises(
        self, coral_1x1: Coral
    ):
        with pytest.raises(ValueError) as e_info:
            coral_1x1.initiate_coral_morphology(np.array([4, 2]))
        assert (
            str(e_info.value)
            == "Spatial dimension of cover does not match: (2,) =/= 1."
        )

    def test_initiate_coral_morphology_with_cover(self, coral_1x1: Coral):
        coral_1x1.initiate_coral_morphology(np.array([4]))


class TestCoral2x2:
    """
    Legacy tests with a 2x2 DataReshape matrix.
    """

    @pytest.fixture(autouse=True)
    def reshape_2x2(self):
        RESHAPE().spacetime = (2, 2)
        rt = RESHAPE()
        assert rt.spacetime == (2, 2)
        return rt

    @pytest.fixture(autouse=False)
    def mixed_coral(self) -> Coral:
        input_dict = dict(
            RESHAPE=DataReshape((2, 2)),
            constants=Constants(),
            dc=0.2,
            hc=0.1,
            bc=0.2,
            tc=0.1,
            ac=0.2,
            Csp=1,
        )
        return Coral(**input_dict)

    def test_input_multiple(self, coral_2x2: Coral):
        for i in range(coral_2x2.RESHAPE.space):
            assert coral_2x2.dc[i] == 0.2
            assert coral_2x2.hc[i] == 0.3
            assert coral_2x2.bc[i] == 0.1
            assert coral_2x2.tc[i] == 0.15
            assert coral_2x2.ac[i] == 0.3

    def test_auto_initiate1(self, coral_2x2: Coral):
        coral_2x2.initiate_coral_morphology()
        for i in range(coral_2x2.RESHAPE.space):
            assert coral_2x2.dc[i] == 0.2
            assert coral_2x2.hc[i] == 0.3
            assert coral_2x2.bc[i] == 0.1
            assert coral_2x2.tc[i] == 0.15
            assert coral_2x2.ac[i] == 0.3

    def test_auto_initiate2(self, coral_2x2: Coral):
        coral_2x2.initiate_coral_morphology(cover=[1, 1])
        for i in range(coral_2x2.RESHAPE.space):
            assert coral_2x2.dc[i] == 0.2
            assert coral_2x2.hc[i] == 0.3
            assert coral_2x2.bc[i] == 0.1
            assert coral_2x2.tc[i] == 0.15
            assert coral_2x2.ac[i] == 0.3

    def test_auto_initiate3(self, coral_2x2: Coral):
        coral_2x2.initiate_coral_morphology(cover=[1, 0])
        assert coral_2x2.dc[0] == 0.2
        assert coral_2x2.hc[0] == 0.3
        assert coral_2x2.bc[0] == 0.1
        assert coral_2x2.tc[0] == 0.15
        assert coral_2x2.ac[0] == 0.3
        assert coral_2x2.dc[1] == 0
        assert coral_2x2.hc[1] == 0
        assert coral_2x2.bc[1] == 0
        assert coral_2x2.tc[1] == 0
        assert coral_2x2.ac[1] == 0

    def test_representative_diameter(self, coral_2x2: Coral):
        coral_2x2.initiate_coral_morphology()
        for i in range(coral_2x2.RESHAPE.space):
            assert float(coral_2x2.dc_rep[i]), pytest.approx(0.15)

    def test_morphological_ratios(self, mixed_coral: dict):
        mixed_coral.initiate_coral_morphology()
        for i in range(mixed_coral.RESHAPE.space):
            assert float(mixed_coral.rf[i]), pytest.approx(1.5)
            assert float(mixed_coral.rp[i]), pytest.approx(0.5)
            assert float(mixed_coral.rs[i]), pytest.approx(2 / 3)

    def test_coral_volume(self, mixed_coral: Coral):
        mixed_coral.initiate_coral_morphology()
        for i in range(mixed_coral.RESHAPE.space):
            assert float(mixed_coral.volume[i]), pytest.approx(0.0058904862254808635)

    def test_vegetation_density(self, mixed_coral: Coral):
        mixed_coral.initiate_coral_morphology()
        for i in range(mixed_coral.RESHAPE.space):
            assert mixed_coral.as_vegetation_density[i], pytest.approx(
                3.3333333333333335
            )

    def test_coral_cover(self, mixed_coral: Coral):
        mixed_coral.initiate_coral_morphology()
        for i in range(mixed_coral.RESHAPE.space):
            assert mixed_coral.cover[i] == 1
