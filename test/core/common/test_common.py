from src.core.common.singletons import CommonConstants
from src.core.common.constants import Constants
from src.core.common.environment import Environment


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
