from typing import Dict, Optional, Protocol, Union, runtime_checkable

import numpy as np

from src.biota_models.mangroves.model.mangrove_constants import MangroveConstants


@runtime_checkable
class MangroveProtocol(Protocol):
    """
    Protocol for all Mangroves to be used.
    """

    @property
    def constants(self) -> MangroveConstants:
        """
        Constants associated to the Vegetation Model to be run.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            Constants: Instance of constants.
        """
        raise NotImplementedError

    def update_mangrove_characteristics(
        self, height, stem_dia, age, cover
    ):
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

