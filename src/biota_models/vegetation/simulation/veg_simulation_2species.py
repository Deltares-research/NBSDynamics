from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from tkinter.tix import Tree
from typing import List, Optional, Union

import pandas as pd
from numpy import True_
from pydantic import validator
from tqdm import tqdm

from src.biota_models.vegetation.bio_process.veg_colonisation import Colonization
from src.biota_models.vegetation.bio_process.veg_hydro_morphodynamics import (
    Hydro_Morphodynamics,
)
from src.biota_models.vegetation.bio_process.veg_mortality import Veg_Mortality
from src.biota_models.vegetation.model.veg_constants import VegetationConstants
from src.biota_models.vegetation.model.veg_model import Vegetation
from src.biota_models.vegetation.output.veg_output_wrapper import VegOutputWrapper
from src.core import RESHAPE
from src.core.simulation.biota_wrapper import BiotaWrapper
from src.core.simulation.multiplebiota_base_simulation import (
    MultipleBiotaBaseSimulation,
)


class VegetationBiotaWrapper(BiotaWrapper):
    biota: Optional[Vegetation]
    output: Optional[VegOutputWrapper]

    @validator("biota", pre=True, allow_reuse=True)
    @classmethod
    def validate_vegetation(
        cls, field_value: Union[dict, Vegetation], values: dict
    ) -> Vegetation:
        """
        Initializes vegetation in case a dictionary is provided. Ensuring the constants are also
        given to the object.
        Args:
            field_value (Union[dict, Vegetation]): Value given by the user for the Vegetation field.
            values (dict): Dictionary of remaining user-given field values.
        Returns:
            Vegetation: Validated instance of 'Vegetation'.
        """
        if isinstance(field_value, Vegetation):
            return field_value
        if isinstance(field_value, dict):
            # Check if constants present in the dictionary:
            if "constants" in field_value.keys():
                # It will be generated automatically.
                # in case parameters are missing an error will also be displayed.
                return Vegetation(**field_value)
            if "constants" in values.keys():
                field_value["constants"] = values["constants"]
                return Vegetation(**field_value)
            raise ValueError(
                "Constants should be provided to initialize a Vegetation Model."
            )
        raise NotImplementedError(f"Validator not available for {type(field_value)}")


class _VegetationSimulation_2species(MultipleBiotaBaseSimulation, ABC):
    """
    Implements the `SimulationProtocol`.
    Facade class that can be implemented through an Adapter pattern.
    VegetationModel simulation.
    """

    # Other fields.
    constants: Optional[VegetationConstants]
    biota_wrapper_list: List[VegetationBiotaWrapper] = []

    @validator("constants", pre=True, allow_reuse=True)
    @classmethod
    def validate_constants(
        cls, field_value: Union[str, Path, VegetationConstants]
    ) -> VegetationConstants:
        """
        Validates the user-input constants value and transforms in case it's a filepath (str, Path).
        Args:
            field_value (Union[str, Path, Constants]): Value given by the user representing Constants.
        Raises:
            NotImplementedError: When the input value does not have any converter.
        Returns:
            Constants: Validated constants value.
        """
        if isinstance(field_value, VegetationConstants):
            return field_value
        if isinstance(field_value, str):
            field_value = Path(field_value)
        if isinstance(field_value, Path):
            return VegetationConstants.from_input_file(field_value)
        raise NotImplementedError(f"Validator not available for {type(field_value)}")

    @validator("biota_wrapper_list", pre=True, each_item=True, allow_reuse=True)
    def validate_each_biota_wrapper(
        cls, value: Union[dict, VegetationBiotaWrapper], values: Optional[dict]
    ) -> VegetationBiotaWrapper:
        """
        Validate each provided biota is valid and in case no explicit constant is provided the one from this simulation will be used.
        Args:
            value (Union[dict, VegetationBiotaWrapper]): Value representing a BiotaWrapper
            values (Optional[dict]): Values already defined in this simulation.
        Returns:
            VegetationBiotaWrapper: Generated BiotaWrapper from the input value.
        """
        if isinstance(value, VegetationBiotaWrapper):
            return value
        if isinstance(value, dict):
            # Include the current constant values if they are missing
            biota_dict: dict = value.get("biota", dict())
            if not "constants" in biota_dict.keys() and "constants" in values.keys():
                biota_dict["constants"] = values["constants"]
                return VegetationBiotaWrapper(**dict(biota=biota_dict))
            return VegetationBiotaWrapper(**value)
        # If we get into this point it will fail with a default (expected) error.
        return value

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

    def initiate(
        self,
        x_range: Optional[tuple] = None,
        y_range: Optional[tuple] = None,
        cover: Optional[Path] = None,
    ) -> Vegetation:
        """Initiate the vegetation distribution.
        The default vegetation distribution is no initial vegetation cover.
        :param x_range: minimum and maximum x-coordinate, defaults to None
        :param y_range: minimum and maximum y-coordinate, defaults to None
        :param cover: veg cover, defaults to None
        :type veg: Vegetation
        :type x_range: tuple, optional
        :type y_range: tuple, optional
        :type cover: Path, optional
        :return: vegetation characteristics initiated
        :rtype: Vegetation
        """

        self.configure_hydrodynamics()
        self.configure_output()
        # Load constants and validate environment.
        self.validate_simulation_directories()

        RESHAPE().space = self.hydrodynamics.space
        xy = self.hydrodynamics.xy_coordinates

        # cover = np.zeros(RESHAPE().space)
        # if x_range is not None:
        #     x_min = x_range[0] if x_range[0] is not None else min(xy[:][0])
        #     x_max = x_range[1] if x_range[1] is not None else max(xy[:][0])
        #     cover[np.logical_or(xy[:][0] <= x_min, xy[:][0] >= x_max)] = 0
        #
        # if y_range is not None:
        #     y_min = y_range[0] if y_range[0] is not None else min(xy[:][1])
        #     y_max = y_range[1] if y_range[1] is not None else max(xy[:][1])
        #     cover[np.logical_or(xy[:][1] <= y_min, xy[:][1] >= y_max)] = 0
        def initiate_biotas(biota_wrapper: VegetationBiotaWrapper):
            # Initiate input
            biota_wrapper.biota.initial.initiate_vegetation_characteristics(cover)
            biota_wrapper.biota.juvenile.initiate_vegetation_characteristics(cover)
            biota_wrapper.biota.mature.initiate_vegetation_characteristics(cover)
            # Initiate output
            if biota_wrapper.output.defined:
                biota_wrapper.output.initialize(biota_wrapper.biota)
            else:
                print("WARNING: No output defined, so none exported.")
            # TODO: Is it really needed to re-initialize the output wrapper?
            biota_wrapper.output.initialize(biota_wrapper.biota)

        for biota_wrapper in self.biota_wrapper_list:
            initiate_biotas(biota_wrapper)

    def run(self, duration: Optional[int] = None):
        """Run simulation.
        :param biota: vegetation
        :param duration: simulation duration [yrs], defaults to None
        :type biota: Vegetation
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

        first_biota: Vegetation = self.biota_wrapper_list[0].biota
        second_biota: Vegetation = self.biota_wrapper_list[1].biota

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
                    ):  # if time_step is input in s! #call hydromorphodynamics every time step and store values to get min
                        # if-statement that encompasses all for which the hydrodynamic should be used

                        progress.set_postfix(inner_loop="update hydrodynamics BMI-Wrapper", ets=str(ets))


                        (
                            cur_tau,
                            cur_vel,
                            cur_wl,
                            bed_level,
                            ba
                        ) = self.hydrodynamics.update_hydromorphodynamics(
                            veg_species1=first_biota,
                            time_step=time_step,
                            veg_species2=second_biota,  # every timestep
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
                            veg=first_biota,
                        )

                        hydro_mor2 = Hydro_Morphodynamics(
                            tau_cur=cur_tau,
                            u_cur=cur_vel,
                            wl_cur=cur_wl,
                            bl_cur=bed_level,
                            ts=ts,
                            veg=second_biota,
                        )

                    hydro_mor.get_hydromorph_values(first_biota)
                    hydro_mor2.get_hydromorph_values(second_biota)

                    # # vegetation dynamics
                    progress.set_postfix(inner_loop="vegetation dynamics")
                    # vegetation mortality and growth update
                    mort = Veg_Mortality
                    mort.update(
                        mort,
                        first_biota,
                        first_biota.constants,
                        ets,
                        begin_date,
                        end_date,
                        period,
                    )
                    mort2 = Veg_Mortality
                    mort2.update(
                        mort,
                        second_biota,
                        second_biota.constants,
                        ets,
                        begin_date,
                        end_date,
                        period,
                    )

                    colstart_species1 = pd.to_datetime(first_biota.constants.ColStart).replace(
                        year=begin_date.year
                    )
                    colend_species1 = pd.to_datetime(first_biota.constants.ColEnd).replace(
                        year=begin_date.year
                    )
                    colstart_species2 = pd.to_datetime(second_biota.constants.ColStart).replace(
                        year=begin_date.year
                    )
                    colend_species2 = pd.to_datetime(second_biota.constants.ColEnd).replace(
                        year=begin_date.year
                    )
                    # # colonization (only in colonization period)
                    # if self.constants.col_days[ets] > 0:
                    if any(colstart_species1 <= pd.to_datetime(period)) and any(
                        pd.to_datetime(period) <= colend_species1
                    ) and any(colstart_species2 <= pd.to_datetime(period)) and any(
                        pd.to_datetime(period) <= colend_species2
                    ):
                        progress.set_postfix(inner_loop="vegetation colonization")
                        col = Colonization()
                        col.update(first_biota, second_biota)

                    elif any(colstart_species1 <= pd.to_datetime(period)) and any(
                        pd.to_datetime(period) <= colend_species1):
                        progress.set_postfix(inner_loop="vegetation colonization")
                        col = Colonization()
                        col.update(first_biota)

                    elif any(colstart_species2 <= pd.to_datetime(period)) and any(
                        pd.to_datetime(period) <= colend_species2):
                        progress.set_postfix(inner_loop="vegetation colonization")
                        col = Colonization()
                        col.update(second_biota)


                    # update lifestages, initial to juvenile and juvenile to mature
                    first_biota.update_lifestages()
                    second_biota.update_lifestages()

                    # # export results
                    progress.set_postfix(inner_loop="export results")
                    # map-file
                    # self.output.map_output.update(self.veg, years[i]) #change to period we are in current ets

                    def update_biotawrapper_map_output(
                        biota_wrapper: VegetationBiotaWrapper,
                    ):
                        biota_wrapper.output.map_output.update(
                            biota_wrapper.biota,
                            int(period[-1].strftime("%Y%m%d")),
                            ets,
                            i,
                            biota_wrapper.biota.constants,
                        )  # change to period we are in current ets
                        # his-file
                        period_days = [
                            begin_date + timedelta(n)
                            for n in range(int((end_date - begin_date).days))
                        ]
                        biota_wrapper.output.his_output.update(
                            biota_wrapper.biota,
                            pd.DataFrame(period_days),
                        )

                    for biota_wrapper in self.biota_wrapper_list:
                        update_biotawrapper_map_output(biota_wrapper)

                    hydro_mor.store_hydromorph_values(first_biota)
                    hydro_mor2.store_hydromorph_values(second_biota)

    def finalise(self):
        """Finalise simulation."""
        self.hydrodynamics.finalise()
