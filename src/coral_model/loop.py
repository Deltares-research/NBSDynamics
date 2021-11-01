"""
coral_model - loop

@author: Gijs G. Hendrickx

"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from src.coral_model import core
from src.coral_model.core import (
    Calcification,
    Dislodgement,
    Flow,
    Light,
    Morphology,
    Photosynthesis,
    PopulationStates,
    Recruitment,
    Temperature,
)
from src.coral_model.environment import Constants, Environment
from src.coral_model.hydrodynamics import Delft3D, Transect  # ,Reef0D,Reef1D
from src.coral_model.utils import Output, time_series_year


class Simulation:
    """CoralModel simulation."""

    working_dir: Optional[Path]

    def __init__(self, mode=None):

        """Simulation loop of a coral model.
        :param mode: mode of hydrodynamics to be used in simulation
        :type mode: str
        """
        self._environment = Environment()
        self._constants = Constants()
        self.working_dir = Path.cwd()
        self.output = None

        modes = {
            #                 'Reef0D'   : Reef0D(),
            #                 'Reef1D'   : Reef1D(),
            "Delft3D": Delft3D(),
            "Transect": Transect(),
        }
        modeset = modes.get(mode, "error")
        if not modeset == "error":
            self._hydrodynamics = modeset
        else:
            msg = f"{mode} not in {modes}."
            raise ValueError(msg)

    @property
    def environment(self):
        """Environment attribute of the Simulation

        :rtype: Environment
        """
        return self._environment

    @property
    def constants(self):
        """Constants attribute of the Simulation

        :rtype: Constants
        """
        return self._constants

    @property
    def hydrodynamics(self):
        """Hydrodynamics attribute of the Simulation

        :rtype: Hydrodynamics
        """
        return self._hydrodynamics

    @property
    def figures_dir(self):
        """Figures directory.

        :rtype: str
        """
        return self.working_dir / "figures"

    @property
    def output_dir(self):
        """Output directory.

        :rtype: str
        """
        return self.working_dir / "output"

    @property
    def input_dir(self):
        """Input directory.

        :rtype: str
        """
        return self.working_dir / "input"

    def set_directories(self, workdir: Path):
        """Set directories based on working directory.

        :param working_dir: working directory
        :param input_dir: input directory

        :type working_dir: str
        :type input_dir: str
        """
        self.working_dir = workdir
        self._make_directories()

    def _make_directories(self):
        """Create directories if not existing."""
        loop_dirs: List[Path] = [
            self.working_dir,
            self.output_dir,
            self.input_dir,
            self.figures_dir,
        ]
        for loop_dir in loop_dirs:
            if not loop_dir.is_dir():
                loop_dir.mkdir(parents=True)

    def read_parameters(self, file="coral_input.txt", folder=None):
        ddir = self.input_dir if folder is None else folder
        infil = os.path.join(ddir, file)
        self._constants.read_it(infil)

    def set_delft3d_environment(self):
        """Set directories and files of hydrodynamic mode 'Delft3D'."""
        # TODO: Set D3D-files and -directories

    def define_output(
        self,
        output_type,
        lme=True,
        fme=True,
        tme=True,
        pd=True,
        ps=True,
        calc=True,
        md=True,
    ):
        """Initiate output files based on requested output data.

        :param output_type: mapping or history output
        :param lme: light micro-environment, defaults to True
        :param fme: flow micro-environment, defaults to True
        :param tme: thermal micro-environment, defaults to True
        :param pd: photosynthetic dependencies, defaults to True
        :param ps: population states, defaults to True
        :param calc: calcification rates, defaults to True
        :param md: morphological development, defaults to True

        :type output_type: str
        :type lme: bool, optional
        :type fme: bool, optional
        :type tme: bool, optional
        :type pd: bool, optional
        :type ps: bool, optional
        :type calc: bool, optional
        :type md: bool, optional
        """
        types = ("map", "his")
        if output_type not in types:
            msg = f"{output_type} not in {types}."
            raise ValueError(msg)

        if not isinstance(self.output, Output):
            self.output = Output(
                self.output_dir,
                self.hydrodynamics.xy_coordinates,
                self.hydrodynamics.outpoint,
                self.environment.dates[0],
            )

        self.output.define_output(
            output_type=output_type,
            lme=lme,
            fme=fme,
            tme=tme,
            pd=pd,
            ps=ps,
            calc=calc,
            md=md,
        )

    def input_check(self):
        """Check input; if all required data is provided."""
        if self.environment.light is None:
            msg = f"CoralModel simulation cannot run without data on light conditions."
            raise ValueError(msg)

        if self.environment.temperature is None:
            msg = f"CoralModel simulation cannot run without data on temperature conditions."
            raise ValueError(msg)

        if self.environment.light_attenuation is None:
            self.environment.set_parameter_values(
                "light_attenuation", self.constants.Kd0
            )
            print(
                f"Light attenuation coefficient set to default: Kd = {self.constants.Kd0} [m-1]"
            )

        if self.environment.aragonite is None:
            self.environment.set_parameter_values("aragonite", self.constants.omegaA0)
            print(
                f"Aragonite saturation state set to default: omega_a0 = {self.constants.omegaA0} [-]"
            )

        # TODO: add other dependencies based on process switches in self.constants if required

    def initiate(self, coral, x_range=None, y_range=None, value=None):
        """Initiate the coral distribution. The default coral distribution is a full coral cover over the whole domain.
        More complex initial conditions of the coral cover cannot be realised with this method. See the documentation on
        workarounds to achieve this anyway.

        :param coral: coral animal
        :param x_range: minimum and maximum x-coordinate, defaults to None
        :param y_range: minimum and maximum y-coordinate, defaults to None
        :param value: coral cover, defaults to None

        :type coral: Coral
        :type x_range: tuple, optional
        :type y_range: tuple, optional
        :type value: float, optional

        :return: coral animal initiated
        :rtype: Coral
        """
        self.input_check()

        #        self.hydrodynamics.initiate()
        core.RESHAPE.space = self.hydrodynamics.space

        if self.output.defined:
            self.output.initiate_his()
            self.output.initiate_map(coral)
        else:
            msg = f"WARNING: No output defined, so none exported."
            print(msg)

        xy = self.hydrodynamics.xy_coordinates

        if value is None:
            value = 1

        cover = value * np.ones(core.RESHAPE.space)

        if x_range is not None:
            x_min = x_range[0] if x_range[0] is not None else min(xy[:][0])
            x_max = x_range[1] if x_range[1] is not None else max(xy[:][0])
            cover[np.logical_or(xy[:][0] <= x_min, xy[:][0] >= x_max)] = 0

        if y_range is not None:
            y_min = y_range[0] if y_range[0] is not None else min(xy[:][1])
            y_max = y_range[1] if y_range[1] is not None else max(xy[:][1])
            cover[np.logical_or(xy[:][1] <= y_min, xy[:][1] >= y_max)] = 0

        coral.initiate_spatial_morphology(cover)

        self.output.initiate_his()
        self.output.initiate_map(coral)

        return coral

    def exec(self, coral, duration=None):
        """Execute simulation.

        :param coral: coral animal
        :param duration: simulation duration [yrs], defaults to None

        :type coral: Coral
        :type duration: int, optional
        """
        # auto-set duration based on environmental time-series
        if duration is None:
            duration = int(
                self.environment.dates.iloc[-1].year
                - self.environment.dates.iloc[0].year
            )
        years = range(
            int(self.environment.dates.iloc[0].year),
            int(self.environment.dates.iloc[0].year + duration),
        )

        with tqdm(range((int(duration)))) as progress:
            for i in progress:
                # set dimensions (i.e. update time-dimension)
                core.RESHAPE.time = len(
                    self.environment.dates.dt.year[
                        self.environment.dates.dt.year == years[i]
                    ]
                )

                # if-statement that encompasses all for which the hydrodynamic should be used
                progress.set_postfix(inner_loop=f"update {self.hydrodynamics}")
                current_vel, wave_vel, wave_per = self.hydrodynamics.update(
                    coral, stormcat=0
                )

                # # environment
                progress.set_postfix(inner_loop="coral environment")
                # light micro-environment
                lme = Light(
                    constants=self.constants,
                    light_in=time_series_year(self.environment.light, years[i]),
                    lac=time_series_year(self.environment.light_attenuation, years[i]),
                    depth=self.hydrodynamics.water_depth,
                )
                lme.rep_light(coral)
                # flow micro-environment
                fme = Flow(
                    constants=self.constants,
                    u_current=current_vel,
                    u_wave=wave_vel,
                    h=self.hydrodynamics.water_depth,
                    peak_period=wave_per,
                )
                fme.velocities(coral, in_canopy=self.constants.fme)
                fme.thermal_boundary_layer(coral)
                # thermal micro-environment
                tme = Temperature(
                    constants=self.constants,
                    temperature=time_series_year(
                        self.environment.temp_kelvin, years[i]
                    ),
                )
                tme.coral_temperature(coral)

                # # physiology
                progress.set_postfix(inner_loop="coral physiology")
                # photosynthetic dependencies
                phd = Photosynthesis(
                    constants=self.constants,
                    light_in=time_series_year(self.environment.light, years[i]),
                    first_year=True if i == 0 else False,
                )
                phd.photo_rate(coral, self.environment, years[i])
                # population states
                ps = PopulationStates(constants=self.constants)
                ps.pop_states_t(coral)
                # calcification
                cr = Calcification(constants=self.constants)
                cr.calcification_rate(
                    coral, time_series_year(self.environment.aragonite, years[i])
                )
                # # morphology
                progress.set_postfix(inner_loop="coral morphology")
                # morphological development
                mor = Morphology(
                    constants=self.constants,
                    calc_sum=coral.calc.sum(axis=1),
                    light_in=time_series_year(self.environment.light, years[i]),
                )
                mor.update(coral)

                # # storm damage
                if self.environment.storm_category is not None:
                    tt = self.environment.storm_category
                    yr = years[i]
                    stormcat = int(tt["stormcat"].values[tt.index == yr])
                    if stormcat > 0:
                        progress.set_postfix(inner_loop="storm damage")
                        # update hydrodynamic model
                        current_vel, wave_vel, wave_per = self.hydrodynamics.update(
                            coral, stormcat
                        )
                        # storm flow environment
                        sfe = Flow(
                            constants=self.constants,
                            u_current=current_vel,
                            u_wave=wave_vel,
                            h=self.hydrodynamics.water_depth,
                            peak_period=wave_per,
                        )
                        sfe.velocities(coral, in_canopy=self.constants.fme)
                        # storm dislodgement criterion
                        sdc = Dislodgement(self.constants)
                        sdc.update(coral)

                # # recruitment
                progress.set_postfix(inner_loop="coral recruitment")
                # recruitment
                rec = Recruitment(self.constants)
                rec.update(coral)

                # # export results
                progress.set_postfix(inner_loop="export results")
                # map-file
                self.output.update_map(coral, years[i])
                # his-file
                self.output.update_his(
                    coral,
                    self.environment.dates[self.environment.dates.dt.year == years[i]],
                )

    def finalise(self):
        """Finalise simulation."""
        self.hydrodynamics.finalise()


# TODO: Define folder structure
#  > working directory
#  > figures directory
#  > input directory
#  > output directory
#  > etc.

# TODO: Model initiation IV: OutputFiles
#  > specify output files (i.e. define file names and directories)
#  > specify model data to be included in output files

# TODO: Model initiation V: initial conditions
#  > specify initial morphology
#  > specify initial coral cover
#  > specify carrying capacity

# TODO: Model simulation I: specify SpaceTime

# TODO: Model simulation II: hydrodynamic module
#  > update hydrodynamics
#  > extract variables

# TODO: Model simulation III: coral environment
#  > light micro-environment
#  > flow micro-environment
#  > temperature micro-environment

# TODO: Model simulation IV: coral physiology
#  > photosynthesis
#  > population states
#  > calcification

# TODO: Model simulation V: coral morphology
#  > morphological development

# TODO: Model simulation VI: storm damage
#  > set variables to hydrodynamic module
#  > update hydrodynamics and extract variables
#  > update coral storm survival

# TODO: Model simulation VII: coral recruitment
#  > update recruitment's contribution

# TODO: Model simulation VIII: return morphology
#  > set variables to hydrodynamic module

# TODO: Model simulation IX: export output
#  > write map-file
#  > write his-file

# TODO: Model finalisation
