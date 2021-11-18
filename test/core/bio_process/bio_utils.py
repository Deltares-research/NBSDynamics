import pytest

from src.core.constants import Constants
from src.core.coral.coral_model import Coral
from src.core.utils import DataReshape


@pytest.fixture(scope="module", autouse=False)
def valid_coral() -> Coral:
    """
    Fixture to generate a valid coral to be used in any test within the bio_process module.

    Returns:
        Coral: Valid Coral object.
    """
    return Coral(
        **dict(
            RESHAPE=DataReshape(),
            constants=Constants(),
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
    return Coral(
        **dict(
            RESHAPE=DataReshape((2, 2)),
            constants=Constants(),
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
        constants=Constants(),
        RESHAPE=DataReshape((2, 2)),
        dc=0.4,
        hc=0.3,
        bc=0.2,
        tc=0.15,
        ac=0.3,
    )
