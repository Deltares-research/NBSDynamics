import pytest

from src.core.constants import Constants
from src.core.coral.coral_model import Coral


@pytest.fixture(scope="module", autouse=False)
def valid_coral() -> Coral:
    """
    Fixture to generate a valid coral to be used in any tes within the bio_process mudle.

    Returns:
        Coral: Valid Coral object.
    """
    return Coral(
        **dict(constants=Constants(), dc=0.2, hc=0.3, bc=0.1, tc=0.15, ac=0.3, Csp=1)
    )
