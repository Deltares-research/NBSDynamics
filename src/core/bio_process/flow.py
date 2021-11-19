import numpy as np
from scipy.optimize import newton

from src.core import RESHAPE
from src.core.common.constants import Constants
from src.core.common.space_time import CoralOnly, DataReshape


class Flow:
    """Flow micro-environment."""

    def __init__(
        self,
        u_current,
        u_wave,
        h,
        peak_period,
        constants: Constants = Constants(),
    ):
        """
        :param u_current: current flow velocity [m s-1]
        :param u_wave: wave flow velocity [m s-1]
        :param h: water depth [m]
        :param peak_period: peak wave period [s]

        :type u_current: float, list, tuple, numpy.ndarray
        :type u_wave: float, list, tuple, numpy.ndarray
        :type h: float, list, tuple, numpy.ndarray
        :type peak_period: float, list, tuple, numpy.ndarray
        """
        _reshape = RESHAPE()
        self.uc = _reshape.variable2array(u_current)
        self.uw = _reshape.variable2array(u_wave)
        self.h = _reshape.variable2matrix(h, "space")
        self.Tp = _reshape.variable2array(peak_period)
        self.active = False if u_current is None and u_wave is None else True
        self.constants = constants

    @property
    def uc_matrix(self):
        """Reshaped current flow velocity."""
        return RESHAPE().variable2matrix(self.uc, "space")

    @property
    def uw_matrix(self):
        """Reshaped wave flow velocity."""
        return RESHAPE().variable2matrix(self.uw, "space")

    def velocities(self, coral, in_canopy=True):
        """In-canopy flow velocities, and depth-averaged flow velocities.

        :param coral: coral animal
        :param in_canopy: determine in-canopy flow (or depth-averaged), defaults to True

        :type coral: Coral
        :type in_canopy: bool, optional
        """
        if self.active:
            alpha_w = np.ones(self.uw.shape)
            alpha_c = np.ones(self.uc.shape)
            if in_canopy:
                idx = coral.volume > 0
                for i in idx:
                    alpha_w[i] = self.wave_attenuation(
                        self.constants,
                        coral.dc_rep[i],
                        coral.hc[i],
                        coral.ac[i],
                        self.uw[i],
                        self.Tp[i],
                        self.h[i],
                        wac_type="wave",
                    )
                    alpha_c[i] = self.wave_attenuation(
                        self.constants,
                        coral.dc_rep[i],
                        coral.hc[i],
                        coral.ac[i],
                        self.uc[i],
                        1e3,
                        self.h[i],
                        wac_type="current",
                    )
            coral.ucm = self.wave_current(alpha_w, alpha_c)
            coral.um = self.wave_current()
        else:
            coral.ucm = 9999 * np.ones(RESHAPE().space)

    def wave_current(self, alpha_w=1, alpha_c=1):
        """Wave-current interaction.

        :param alpha_w: wave-attenuation coefficient, defaults to 1
        :param alpha_c: current-attenuation coefficient, defaults to 1

        :type alpha_w: float, list, tuple, numpy.ndarray, optional
        :type alpha_c: float, list, tuple, numpy.ndarray, optional

        :return: wave-current interaction
        :rtype: float, numpy.ndarray
        """
        return np.sqrt(
            (alpha_w * self.uw) ** 2
            + (alpha_c * self.uc) ** 2
            + 2 * alpha_w * self.uw * alpha_c * self.uc * np.cos(self.constants.wcAngle)
        )

    @staticmethod
    def wave_attenuation(
        constants, diameter, height, distance, velocity, period, depth, wac_type
    ):
        """Wave-attenuation coefficient.

        :param diameter: representative coral diameter [m]
        :param height: coral height [m]
        :param distance: axial distance [m]
        :param velocity: flow velocity [m s-1]
        :param period: wave period [s]
        :param depth: water depth [m]
        :param wac_type: type of wave-attenuation coefficient [-]

        :type diameter: float
        :type height: float
        :type distance: float
        :type velocity: float
        :type depth: float
        :type depth: float
        :type wac_type: str
        """
        # TODO: Split this method in one solely focusing on the wave attenuation coefficient;
        #  and one implementing this method to dynamically determine the drag coefficient.
        #  Thus, reformat this method as in coral_model_v0.
        # # function and derivative definitions
        def function(beta):
            """Complex-valued function to be solved, where beta is the complex representation of the wave-attenuation
            coefficient.
            """
            # components
            shear = (
                (8.0 * above_motion)
                / (3.0 * np.pi * shear_length)
                * (abs(1.0 - beta) * (1.0 - beta))
            )
            drag = (
                (8.0 * above_motion) / (3.0 * np.pi * drag_length) * (abs(beta) * beta)
            )
            inertia = (
                1j * beta * ((constants.Cm * lambda_planar) / (1.0 - lambda_planar))
            )
            # combined
            f = 1j * (beta - 1.0) - shear + drag + inertia
            # output
            return f

        def derivative(beta):
            """Complex-valued derivative to be used to solve the complex-valued function, where beta is the complex
            representation of the wave-attenuation coefficient.
            """
            # components
            shear = (
                (1.0 - beta) ** 2 / abs(1.0 - beta) - abs(1.0 - beta)
            ) / shear_length
            drag = (beta ** 2 / abs(beta) + beta) / drag_length
            inertia = 1j * (constants.Cm * lambda_planar) / (1.0 - lambda_planar)
            # combined
            df = 1j + (8.0 * above_motion) / (3.0 * np.pi) * (-shear + drag) + inertia
            # output
            return df

        # # Input check
        def wave_wac():
            return abs(
                newton(
                    function,
                    x0=complex(0.1, 0.1),
                    fprime=derivative,
                    maxiter=constants.maxiter_aw,
                )
            )

        def current_wac():
            x = drag_length / shear_length * (height / (depth - height) + 1)
            return (x - np.sqrt(x)) / (x - 1)

        wac_type_funcs = dict(current=current_wac, wave=wave_wac)
        wac_function = wac_type_funcs.get(wac_type, None)
        if wac_function is None:
            raise ValueError(f"WAC-type ({wac_type}) not in {wac_type_funcs.keys()}.")

        # # parameter definitions
        # geometric parameters
        planar_area = 0.25 * np.pi * diameter ** 2
        frontal_area = diameter * height
        total_area = 0.5 * distance ** 2
        lambda_planar = planar_area / total_area
        lambda_frontal = frontal_area / total_area
        shear_length = height / (constants.Cs ** 2)
        # # calculations
        wac = 1.0
        if depth <= height:
            return wac

        # If depth > height
        # initial iteration values
        above_flow = velocity
        drag_coefficient = 1.0
        # iteration
        for k in range(int(constants.maxiter_k)):
            drag_length = (2 * height * (1 - lambda_planar)) / (
                drag_coefficient * lambda_frontal
            )
            above_motion = (above_flow * period) / (2 * np.pi)
            wac = wac_function()

            porous_flow = wac * above_flow
            constricted_flow = (
                (1 - lambda_planar)
                / (1 - np.sqrt((4 * lambda_planar) / (constants.psi * np.pi)))
                * porous_flow
            )
            reynolds = (constricted_flow * diameter) / constants.nu
            new_drag = 1 + 10 * reynolds ** (-2.0 / 3)
            if abs((new_drag - drag_coefficient) / new_drag) <= constants.err:
                break
            else:
                drag_coefficient = float(new_drag)
                above_flow = abs(
                    (1 - constants.numericTheta) * above_flow
                    + constants.numericTheta
                    * (depth * velocity - height * porous_flow)
                    / (depth - height)
                )

            if k == constants.maxiter_k:
                print(
                    f"WARNING: maximum number of iterations reached "
                    f"({constants.maxiter_k})"
                )

        return wac

    def thermal_boundary_layer(self, coral):
        """Thermal boundary layer.

        :param coral: coral animal
        :type coral: Coral
        """
        if self.active and self.constants.tme:
            delta = self.velocity_boundary_layer(self.constants, coral)
            coral.delta_t = delta * (
                (self.constants.alpha / self.constants.nu) ** (1 / 3)
            )

    @staticmethod
    def velocity_boundary_layer(constants, coral):
        """Velocity boundary layer.

        :param coral: coral animal
        :type coral: Coral
        """

        def boundary_layer(rd, nu, cf, ucm):
            """Thickness velocity boundary layer."""
            return (rd * nu) / (np.sqrt(cf) * ucm)

        return CoralOnly().in_space(
            coral=coral,
            function=boundary_layer,
            args=(constants.rd, constants.nu, constants.Cf, coral.ucm),
        )
