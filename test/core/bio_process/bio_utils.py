import pytest

from src.core.constants import Constants
from src.core.coral.coral_model import Coral


@pytest.fixture(scope="module", autouse=True)
def valid_coral() -> Coral:
    """
    Fixture to generate a valid coral to be used in any tes within the bio_process mudle.

    Returns:
        Coral: Valid Coral object.
    """
    return Coral(
        **dict(constants=Constants(), dc=0.2, hc=0.1, bc=0.2, tc=0.1, ac=0.2, Csp=1)
    )
