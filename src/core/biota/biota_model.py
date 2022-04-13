from src.core.base_model import ExtraModel
from src.core.common.base_constants import BaseConstants


class Biota(ExtraModel):
    """
    Empty class to cluster all the BIOTA models so that we can reference to this abstraction from protocols and other classes.
    """

    constants: BaseConstants
    pass
