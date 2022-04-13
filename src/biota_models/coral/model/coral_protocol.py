from typing import Dict, Optional, Protocol, Union, runtime_checkable

import numpy as np

from src.biota_models.coral.model.coral_constants import CoralConstants


@runtime_checkable
class CoralProtocol(Protocol):
    """
    Protocol for all Corals to be used in the `NBSDynamics` Project.
    """

    @property
    def constants(self) -> CoralConstants:
        """
        Constants associated to the Coral Model to be run.

        Raises:
            NotImplementedError: When the model does not implement its own definition.

        Returns:
            Constants: Instance of constants.
        """
        raise NotImplementedError

    def initiate_coral_morphology(self, cover: Optional[np.ndarray]):
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

    def update_coral_morphology(
        self,
        coral_volume: Union[float, np.ndarray],
        morphology_ratios: Dict[str, Union[float, np.ndarray]],
    ):
        """
        Update the coral morphology based on updated coral volume and morphology ratios.

        Args:
            coral_volume (Union[float, np.ndarray]): Coral volume
            morphology_ratios (Dict[str, Union[float, np.ndarray]]): Morphology ratios (rf, rp, rs, ..)

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError
