from typing import List

from src.core.hydrodynamics.delft3d import Delft3D, DimrModel, FlowFmModel
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.hydrodynamics.reef_0d import Reef0D
from src.core.hydrodynamics.reef_1d import Reef1D
from src.core.hydrodynamics.transect import Transect


class HydrodynamicsFactory:
    """
    Factory class to select hydrodynamic models.
    It also works as a binding model-protocol between the hydrodynamic models
    and the `HydrodynamicProtocol`.
    """

    supported_modes: List[HydrodynamicProtocol] = [
        Reef0D,
        Reef1D,
        DimrModel,
        FlowFmModel,
        Transect,
    ]

    @staticmethod
    def get_hydrodynamic_model_type(model_name: str) -> HydrodynamicProtocol:
        """
        Returns the type associated with the given model name.

        Args:
            model_name (str): Model name to retrieve.

        Raises:
            ValueError: When the model name is not associated with a valid `HydrodynamicProtocol`.

        Returns:
            HydrodynamicProtocol: The requested model type.
        """
        nm_name = model_name.lower() if model_name is not None else model_name
        hydromodel: HydrodynamicProtocol = next(
            (
                m_type
                for m_type in HydrodynamicsFactory.supported_modes
                if m_type.__name__.lower() == nm_name
            ),
            None,
        )
        if hydromodel is None:

            msg = f"{model_name} not in {[x.__name__ for x in HydrodynamicsFactory.supported_modes]}."
            raise ValueError(msg)
        return hydromodel
