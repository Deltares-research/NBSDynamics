from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from pydantic import validator
from tqdm import tqdm

from src.biota_models.mangroves.bio_process.mangrove_colonisation import Colonization
from src.biota_models.mangroves.bio_process.mangrove_hydro_morphodynamics import (
    Hydro_Morphodynamics,
)
from src.biota_models.mangroves.bio_process.mangrove_mortality import Mangrove_Mortality
from src.biota_models.mangroves.model.mangrove_constants import MangroveConstants
from src.biota_models.mangroves.model.mangrove_model import Mangrove
from src.biota_models.mangroves.output.mangrove_output_wrapper import MangroveOutputWrapper
from src.core import RESHAPE
from src.core.hydrodynamics.factory import HydrodynamicsFactory
from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol
from src.core.simulation.base_simulation import BaseSimulation

class _MangroveSimulatio(BaseSimulation, ABC):
    """
    Implements the `SimulationProtocol`.
    Facade class that can be implemented through an Adapter pattern.
    MangroveModel simulation.
    """

    constants: Optional[MangroveConstants]
    output: Optional[MangroveOutputWrapper]
    biota: Optional[Mangrove]

    @validator("constants", pre=True)
    @classmethod
    def validate_constants(
            cls, field_value: Union[str, Path, MangroveConstants]
    ) -> MangroveConstants:
        """
        Validates the user-input constants value and transforms in case it's a filepath (str, Path).

        Args:
            field_value (Union[str, Path, Constants]): Value given by the user representing Constants.

        Raises:
            NotImplementedError: When the input value does not have any converter.

        Returns:
            Constants: Validated constants value.
        """
        if isinstance(field_value, MangroveConstants):
            return field_value
        if isinstance(field_value, str):
            field_value = Path(field_value)
        if isinstance(field_value, Path):
            return MangroveConstants.from_input_file(field_value)
        raise NotImplementedError(f"Validator not available for {type(field_value)}")

    @validator("biota", pre=True)
    @classmethod
    def validate_mangrove(
        cls, field_value: Union[dict, Mangrove], values: dict
    ) -> Mangrove:
        """
        Initializes vegetation in case a dictionary is provided. Ensuring the constants are also
        given to the object.

        Args:
            field_value (Union[dict, Vegetation]): Value given by the user for the Mangrove field.
            values (dict): Dictionary of remaining user-given field values.

        Returns:
            Mangrove: Validated instance of 'Mangrove'.
        """
        if isinstance(field_value, Mangrove):
            return field_value
        if isinstance(field_value, dict):
            # Check if constants present in the dictionary:
            if "constants" in field_value.keys():
                # It will be generated automatically.
                # in case parameters are missing an error will also be displayed.
                return Mangrove(**field_value)
            if "constants" in values.keys():
                field_value["constants"] = values["constants"]
                return Mangrove(**field_value)
            raise ValueError(
                "Constants should be provided to initialize a Mangrove Model."
            )
        raise NotImplementedError(f"Validator not available for {type(field_value)}")

    @validator("hydrodynamics", pre=True, always=True)
    @classmethod
    def validate_hydrodynamics_present(
        cls, field_values: Union[dict, HydrodynamicProtocol], values: dict
    ) -> HydrodynamicProtocol:
        """
        Validator to transform the given dictionary into the corresponding hydrodynamic model.

        Args:
            field_values (Union[dict, HydrodynamicProtocol]): Value assigned to `hydrodynamics`.
            values (dict): Dictionary of values given by the user.

        Raises:
            ValueError: When no hydrodynamics model can be built with the given values.

        Returns:
            dict: Validated dictionary of values given by the user.
        """
        if field_values is None:
            field_values = dict()
        if isinstance(field_values, dict):
            return HydrodynamicsFactory.create(
                field_values.get("mode", values["mode"]), **field_values
            )

        return field_values

    @abstractmethod
    def configure_hydrodynamics(self):
        """
        Configures the parameters for the `HydrodynamicsProtocol`.

        Raises:
            NotImplementedError: When abstract method not defined in concrete class.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_output(self):
        """
        Configures the parameters for the `OutputWrapper`.
        """
        raise NotImplementedError

    def validate_simulation_directories(self):
        """
        Generates the required directories if they do not exist already.
        """
        loop_dirs: List[Path] = [
            "working_dir",
            "output_dir",
            "input_dir",
            "figures_dir",
        ]
        for loop_dir in loop_dirs:
            value_dir: Path = getattr(self, loop_dir)
            if not value_dir.is_dir():
                value_dir.mkdir(parents=True)

    def initiate(self, cover: Optional [Path] = None) -> Mangrove:
        """Initiate the mangrove distribution.
        The default mangrove distribution is no initial vegetation cover.

        :param cover: initial mangrove cover, defaults to None
        :param x_range: minimum and maximum x-coordinate, defaults to None
        :param y_range: minimum and maximum y-coordinate, defaults to None


        :type veg: Vegetation
        :type cover: Path, optional
        :type x_range: tuple, optional
        :type y_range: tuple, optional


        :return: mangrove characteristics initiated
        :rtype: Mangrove
        """
        self.configure_hydrodynamics()
        self.configure_output()
        # Load constants and validate environment.
        self.validate_simulation_directories()

        RESHAPE().space = self.hydrodynamics.space

        self.biota.initiate_vegetation_characteristics(cover)

        if self.output.defined:
            self.output.initialize(self.biota)
        else:
            print("WARNING: No output defined, so none exported.")

        self.output.initialize(self.biota)

    def run(self, duration: Optional[int] = None):
        """Run simulation.

        :param mang: mangrove
        :param duration: simulation duration [yrs], defaults to None


        :type veg: Vegetation
        :type duration: int, optional

        """

        # auto-set duration based on constants value (provided or default)
        if duration is None:
            duration = int(self.constants.sim_duration)
        start_date = pd.to_datetime(self.constants.start_date)
        years = range(
            int(start_date.year),
            int(start_date.year + duration),
        )  # takes the starting year from the start date defined in the Constants class.

        with tqdm(range((int(duration)))) as progress:
            for i in progress:
                current_year = years[i]
                for ets in range(0, self.constants.t_eco_year):
                    if ets == 0 and i == 0:
                        begin_date = pd.Timestamp(
                            year=current_year,
                            month=start_date.month,
                            day=start_date.day,
                        )
                    else:
                        begin_date = end_date
                    end_date = begin_date + timedelta(
                        days=(365 / self.constants.t_eco_year)
                    )
                    period = [
                        begin_date + timedelta(seconds=n)
                        for n in range(
                            int(
                                (end_date - begin_date).days * 24 * 3600
                                + (end_date - begin_date).seconds
                            )
                        )
                    ]

                    # # set dimensions (i.e. update time-dimension)
                    RESHAPE().time = len(pd.DataFrame(period))

                    time_step = 11178
                    for ts in range(
                        0, len(period), time_step
                    ):  # every quarter of a M2 tidal cycle (12.42 hours) the hydro-morphodynamic information are taken from DFM

                        progress.set_postfix(inner_loop=f"update {self.hydrodynamics}")

                        (
                            cur_tau,
                            cur_vel,
                            cur_wl,
                            bed_level,
                        ) = self.hydrodynamics.update_hydromorphodynamics(
                            self.biota, time_step=time_step  # 4 times every tidal cycle (M2 tide)
                        )

                        # # environment
                        progress.set_postfix(inner_loop="hydromorpho environment")
                        # hydromorpho environment
                        hydro_mor = Hydro_Morphodynamics(
                            tau_cur=cur_tau,
                            u_cur=cur_vel,
                            wl_cur=cur_wl,
                            bl_cur=bed_level,
                            ts=ts,
                            veg=self.biota,
                        )
                    hydro_mor.get_hydromorph_values(self.biota)

                    # # mangrove dynamics
                    progress.set_postfix(inner_loop="vegetation dynamics")

                    # Mortality
                    mort = Mangrove_Mortality
                    mort.update(mort, )

                    # Growth


                    # Colonization
                    col = Colonization()
                    col.update(self.biota)

                    # export results

                    # store hydrdynamics
                    hydro_mor.store_hydromorph_values(self.biota)

    def finalise(self):
        """Finalise simulation."""
        self.hydrodynamics.finalise()