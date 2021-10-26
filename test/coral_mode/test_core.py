import pytest

class TestDummy:

    @pytest.mark.parametrize("a,b,c", [(1,1, 2), (2, 1, 3)])
    def test_given_a_b_returns_expectation(self, a: int, b: int, c: int):
        """
        Dummy test to verify correct functioning / discovering of tests.
        """
        assert a + b == c