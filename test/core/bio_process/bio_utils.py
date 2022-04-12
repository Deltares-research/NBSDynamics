import pytest

from src.core import RESHAPE
from src.core.biota.coral.coral_model import Coral
from src.core.common.coral_constants import CoralConstants
from src.core.common.space_time import DataReshape


@pytest.fixture(scope="module", autouse=True)
def default_constants() -> CoralConstants:
    return CoralConstants()


@pytest.fixture(scope="module", autouse=False)
def matrix_1x1() -> DataReshape:
    RESHAPE().spacetime = (1, 1)
    return RESHAPE()


@pytest.fixture(scope="module", autouse=False)
def matrix_2x2() -> DataReshape:
    RESHAPE().spacetime = (2, 2)
    return RESHAPE()


@pytest.fixture(scope="module", autouse=False)
def valid_coral() -> Coral:
    """
    Fixture to generate a valid coral to be used in any test within the bio_process module.

    Returns:
        Coral: Valid Coral object.
    """
    RESHAPE().spacetime = (1, 1)
    rs = RESHAPE()
    assert rs.spacetime == (1, 1)
    return Coral(
        **dict(
            dc=0.2,
            hc=0.3,
            bc=0.1,
            tc=0.15,
            ac=0.3,
            Csp=1,
        )
    )


@pytest.fixture(scope="module", autouse=False)
def coral_2x2() -> Coral:
    """
    Fixture to generate a valid coral with a DataReshape matrix of 2x2.

    Returns:
        Coral: Coral in a 2x2 matrix.
    """
    RESHAPE().spacetime = (2, 2)
    rs = RESHAPE()
    assert rs.spacetime == (2, 2)
    return Coral(
        **dict(
            dc=0.2,
            hc=0.3,
            bc=0.1,
            tc=0.15,
            ac=0.3,
            Csp=1,
        )
    )


@pytest.fixture(scope="module", autouse=False)
def no_base_coral_2x2() -> Coral:
    return Coral(
        RESHAPE=DataReshape((2, 2)),
        dc=0.4,
        hc=0.3,
        bc=0.2,
        tc=0.15,
        ac=0.3,
    )
