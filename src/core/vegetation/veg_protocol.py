from typing import Dict, Optional, Protocol, Union, runtime_checkable

import numpy as np

from src.core.common.constants_veg import Constants

@runtime_checkable
class VegProtocol(Protocol):
    """
    Protocol for all Vegetation to be used.
    """

    @property
    def constants(self) -> Constants:
        """
        Constants associated to the Vegetation Model to be run.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            Constants: Instance of constants.
        """
        raise NotImplementedError

    def update_vegetation_characteristics_growth(self, veg_height, stem_dia, veg_root, m_height, m_stemdia, m_root):
        """
        Initiate the morphology based on the on set of morphological dimensions and the coral cover. This method
        contains a catch that it can only be used to initiate the morphology, and cannot overwrite existing spatial
        heterogeneous morphology definitions.

        Args:
            cover (Optional[np.ndarray]): Custom coral definition.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError

    def update_vegetation_characteristics_winter(self, veg_height, stem_dia, veg_root):
        """
        Update the coral morphology based on updated coral volume and morphology ratios.

        Args:
            coral_volume (Union[float, np.ndarray]): Coral volume
            morphology_ratios (Dict[str, Union[float, np.ndarray]]): Morphology ratios (rf, rp, rs, ..)

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError


    