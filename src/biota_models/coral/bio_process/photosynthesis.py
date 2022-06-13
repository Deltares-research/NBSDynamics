import numpy as np
import pandas as pd

from src.biota_models.coral.model.coral_constants import CoralConstants
from src.biota_models.coral.model.coral_model import Coral
from src.core import RESHAPE
from src.core.common.space_time import DataReshape


class Photosynthesis:
    """Photosynthesis."""

    def __init__(
        self, light_in, first_year, constants: CoralConstants = CoralConstants()
    ):
        """
        Photosynthetic efficiency based on photosynthetic dependencies.

        :param light_in: incoming light-intensity at the water-air interface [umol photons m-2 s-1]
        :param first_year: first year of the model simulation

        :type light_in: float, list, tuple, numpy.ndarray
        :type first_year: bool
        """
        self.I0 = RESHAPE().variable2matrix(light_in, "time")
        self.first_year = first_year

        self.pld = 1
        self.ptd = 1
        self.pfd = 1

        self.constants = constants

    def photo_rate(self, coral, environment, year):
        """Photosynthetic efficiency.

        :param coral: coral animal
        :param environment: environmental conditions
        :param year: year of simulation

        :type coral: Coral
        :type environment: Environment
        :type year: int
        """
        # components
        self.light_dependency(coral, "qss")
        self.thermal_dependency(coral, environment, year)
        self.flow_dependency(coral)

        # combined
        coral.photo_rate = self.pld * self.ptd * self.pfd

    def light_dependency(self, coral, output):
        """Photosynthetic light dependency.

        :param coral: coral animal
        :param output: type of output

        :type coral: Coral
        :type output: str
        """

        def photo_acclimation(x_old, param):
            """Photo-acclimation."""
            # input check
            params = ("Ik", "Pmax")
            if param not in params:
                message = f"{param} not in {params}."
                raise ValueError(message)

            # parameter definitions
            x_max = self.constants.ik_max if param == "Ik" else self.constants.pm_max
            beta_x = self.constants.betaI if param == "Ik" else self.constants.betaP

            # calculations
            xs = x_max * (coral.light / self.I0) ** beta_x
            if output == "qss":
                return xs
            elif output == "new":
                return xs + (x_old - xs) * np.exp(-self.constants.iota)

        # # parameter definitions
        if output == "qss":
            ik = photo_acclimation(None, "Ik")
            p_max = photo_acclimation(None, "Pmax")
        else:
            msg = f"Only the quasi-steady state solution is currently implemented; use key-word 'qss'."
            raise NotImplementedError(msg)

        # # calculations
        self.pld = p_max * (
            np.tanh(coral.light / ik) - np.tanh(self.constants.Icomp * self.I0 / ik)
        )

    def thermal_dependency(self, coral: Coral, env, year):
        """Photosynthetic thermal dependency.

        :param coral: coral animal
        :param env: environmental conditions
        :param year: year of simulation

        :type coral: Coral
        :type env: Environment
        :type year: int
        """
        _reshape = RESHAPE()

        def thermal_acc():
            """Thermal-acclimation."""
            if self.constants.tme:
                if self.first_year:
                    env.tmeMMMmin = (
                        pd.DataFrame(
                            data=pd.concat(
                                [env.temp_mmm["min"]] * _reshape.space, axis=1
                            ).values,
                            columns=[np.arange(_reshape.space)],
                        )
                        + coral.dTc
                    )
                    env.tmeMMMmax = (
                        pd.DataFrame(
                            data=pd.concat(
                                [env.temp_mmm["max"]] * _reshape.space, axis=1
                            ).values,
                            columns=[np.arange(_reshape.space)],
                        )
                        + coral.dTc
                    )
                else:
                    env.tmeMMMmin[env.tmeMMM.index == year] += coral.dTc
                    env.tmeMMMmax[env.tmeMMm.index == year] += coral.dTc

                mmm_min = env.tmeMMMmin[
                    np.logical_and(
                        env.tmeMMM.index < year,
                        env.tmeMMM.index >= year - int(self.constants.nn / coral.Csp),
                    )
                ]
                m_min = mmm_min.mean(axis=0)
                s_min = mmm_min.std(axis=0)

                mmm_max = env.tmeMMMmax[
                    np.logical_and(
                        env.tmeMMM.index < year,
                        env.tmeMMM.index >= year - int(self.constants.nn / coral.Csp),
                    )
                ]
                m_max = mmm_max.mean(axis=0)
                s_max = mmm_max.std(axis=0)
            else:
                mmm = env.temp_mmm[
                    np.logical_and(
                        env.temp_mmm.index < year,
                        env.temp_mmm.index >= year - int(self.constants.nn / coral.Csp),
                    )
                ]
                m_min, m_max = mmm.mean(axis=0)
                s_min, s_max = mmm.std(axis=0)

            coral.Tlo = m_min - self.constants.k_var * s_min
            coral.Thi = m_max + self.constants.k_var * s_max

        def adapted_temp():
            """Adapted temperature response."""

            def spec():
                """Specialisation term."""
                return 4e-4 * np.exp(-0.33 * (delta_temp - 10))

            response = -(coral.temp - coral.Tlo) * (
                (coral.temp - coral.Tlo) ** 2 - delta_temp**2
            )
            temp_cr = coral.Tlo - (1 / np.sqrt(3)) * delta_temp
            try:
                if self.constants.tme:
                    response[coral.temp <= temp_cr] = -(
                        (2 / (3 * np.sqrt(3))) * delta_temp[coral.temp <= temp_cr] ** 3
                    )
                else:
                    response[coral.temp <= temp_cr] = -(
                        (2 / (3 * np.sqrt(3))) * delta_temp**3
                    )
            except TypeError:
                if coral.temp <= temp_cr:
                    response = (2 / (3 * np.sqrt(3))) * delta_temp**3

            return response * spec()

        def thermal_env():
            """Thermal envelope."""
            return np.exp(
                (self.constants.Ea / self.constants.R) * (1 / 300 - 1 / temp_opt)
            )

        # # parameter definitions
        thermal_acc()
        delta_temp = coral.Thi - coral.Tlo
        temp_opt = coral.Tlo + (1 / np.sqrt(3)) * delta_temp

        # # calculations
        f1 = adapted_temp()
        f2 = thermal_env()
        self.ptd = f1 * f2

    def flow_dependency(self, coral: Coral):
        """Photosynthetic flow dependency.

        :param coral: coral animal
        :type coral: Coral
        """
        if self.constants.pfd:
            pfd = self.constants.pfd_min + (1 - self.constants.pfd_min) * np.tanh(
                2 * coral.ucm / self.constants.ucr
            )
            self.pfd = RESHAPE().variable2matrix(pfd, "space")
        else:
            self.pfd = 1
