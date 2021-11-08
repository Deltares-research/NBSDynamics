"""
coral_model v3 - hydrodynamics

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""
import faulthandler
from typing import Dict, List

from src.core.hydrodynamics.delft3d import Delft3D
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.hydrodynamics.reef_0d import Reef0D
from src.core.hydrodynamics.reef_1d import Reef1D
from src.core.hydrodynamics.transect import Transect

faulthandler.enable()


class HydrodynamicsFactory:
    """
    Factory class to select hydrodynamic models.
    It also works as a binding model-protocol between the hydrodynamic models
    and the 'HydrodynamicProtocol'.
    """

    @staticmethod
    def get_hydrodynamic_model(model_name: str) -> HydrodynamicProtocol:
        modes: List[HydrodynamicProtocol] = [Reef0D, Reef1D, Delft3D, Transect]
        nm_name = model_name.lower() if model_name is not None else model_name
        hydromodel: HydrodynamicProtocol = next(
            (m_type for m_type in modes if m_type.__name__.lower() == nm_name), None
        )
        if hydromodel is None:

            msg = f"{model_name} not in {[x.__name__ for x in modes]}."
            raise ValueError(msg)
        return hydromodel()


class Hydrodynamics:
    """Interface for all hydrodynamic model modes."""

    def set_files(self, mdu=None, config=None):
        """Set critical files of hydrodynamic model.

        :param mdu: MDU-file
        :param config: config-file

        :type mdu: str
        :type config: str
        """
        [self.__set_file(key, val) for key, val in locals().items()]

    def __set_file(self, obj, file):
        """Set file of hydrodynamic model.

        :param obj: file-object to be defined
        :param file: file

        :type obj: str
        :type file: str
        """
        if file is not None and hasattr(self.model, obj):
            setattr(self.model, obj, file)

    def set_update_intervals(self, default, storm=None):
        """Set update intervals; required for Delft3D-model.

        :param default: default update interval
        :param storm: storm update interval, defaults to None

        :type default: int
        :type storm: int, optional
        """
        self.__model.update_interval = default
        self.__model.update_interval_storm = default if storm is None else storm

        if not isinstance(self.model, Delft3D):
            print(
                f"INFO: Update intervals unused; {self.mode} does not use update intervals."
            )

    def input_check(self):
        """Check if all requested content is provided, depending on the mode chosen."""
        _ = self.xy_coordinates
        _ = self.water_depth

        if isinstance(self.model, Delft3D):
            self.input_check_extra_d3d()

    def input_check_extra_d3d(self):
        """Delft3D-specific input check."""
        files = ("mdu",)
        [self.input_check_definition(file) for file in files]

        interval_types = ("update_interval", "update_interval_storm")
        [self.input_check_definition(interval) for interval in interval_types]

    def input_check_definition(self, obj):
        """Check definition of critical object."""
        if getattr(self.model, obj) is None:
            msg = f"{obj} undefined (required for {self.mode}-mode)"
            raise ValueError(msg)
