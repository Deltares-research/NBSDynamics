from _pytest.fixtures import fixture

from src.core.common.constants import _Constants as Constants
from src.core.common.singletons import RESHAPE, CommonConstants
from src.core.common.space_time import DataReshape


class TestCommonConstants:
    def test_commonconstants_singleton(self):
        const_0 = CommonConstants()
        const_1 = CommonConstants()
        assert const_0 is const_1
        assert isinstance(const_0, CommonConstants)
        assert issubclass(type(const_0), Constants)
        # Verify changing variables work.
        assert const_0.tme == const_1.tme
        const_0.tme = not const_0.tme
        assert const_0.tme == const_1.tme


class TestRESHAPE:
    @fixture(autouse=True)
    def reset_RESHAPE(self):
        # Force reshape to initiate.
        RESHAPE._instance = None

    def test_RESHAPE_as_singleton(self):
        const_0 = RESHAPE()
        const_1 = RESHAPE()
        assert const_0 is const_1
        assert isinstance(const_0, RESHAPE)
        assert issubclass(type(const_0), DataReshape)
        # Verify changing variables work.
        assert const_0.spacetime == (1, 1)
        const_1.spacetime = (2, 2)
        assert const_0.spacetime == (2, 2)

    def test_RESHAPE_init(self):
        # Force reshape to initiate.
        RESHAPE._instance = None
        r_single = RESHAPE(spacetime=(2, 2))
        assert r_single.spacetime == (2, 2)
