from typing import Dict, Protocol, Optional, Union
import numpy as np


class CoralProtocol(Protocol):
    """
    Protocol for all Corals to be used in the `NBSDynamics` Project.
    """

    def initate_coral_morphology(self, cover: Optional[np.ndarray]):
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
