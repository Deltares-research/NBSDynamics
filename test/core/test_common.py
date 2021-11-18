from src.core.common import CommonConstants, CommonEnvironment
from src.core.constants import Constants
from src.core.environment import Environment


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


class TestCommonEnvironment:
    def test_commonenvironment_singleton(self):
        const_0 = CommonEnvironment()
        const_1 = CommonEnvironment()
        assert const_0 is const_1
        assert isinstance(const_0, CommonEnvironment)
        assert issubclass(type(const_0), Environment)
