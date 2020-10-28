"""
CoralModel v3 - utils

@author: Gijs G. Hendrickx
"""

import numpy as np


class SpaceTime:
    """Spacetime-object, which validates the definition of the spacetime dimensions."""

    __spacetime = None

    def __init__(self, spacetime=None):
        """
        :param spacetime: spacetime dimensions, defaults to None
        :type spacetime: None, tuple, optional
        """
        if spacetime is not None:
            self.spacetime = spacetime

    def __repr__(self):
        """Development representation."""
        return f'SpaceTime({self.__spacetime})'

    def __str__(self):
        """Print representation."""
        return str(self.spacetime)

    @property
    def spacetime(self):
        """Spacetime dimensions.

        :rtype: tuple
        """
        if self.__spacetime is None:
            return 1, 1
        return self.__spacetime

    @spacetime.setter
    def spacetime(self, space_time):
        """
        :param space_time: spacetime dimensions
        :type space_time: tuple, list, numpy.ndarray
        """
        if not isinstance(space_time, (tuple, list, np.ndarray)):
            msg = f'spacetime must be of type tuple, {type(space_time)} is given.'
            raise TypeError(msg)

        if not len(space_time) == 2:
            msg = f'spacetime must be of size 2, {len(space_time)} is given.'
            raise ValueError(msg)

        if not all(isinstance(dim, int) for dim in space_time):
            msg = f'spacetime must consist of integers only, {[type(dim) for dim in space_time]} is given.'
            raise TypeError(msg)

        self.__spacetime = tuple(space_time)

    @property
    def space(self):
        """Space dimension.

        :rtype: int
        """
        return self.spacetime[0]

    @space.setter
    def space(self, x):
        """
        :param x: space dimension
        :type x: int
        """
        self.spacetime = (x, self.time)

    @property
    def time(self):
        """Time dimension.

        :rtype: int
        """
        return self.spacetime[1]

    @time.setter
    def time(self, t):
        """
        :param t: time dimension
        :type t: int
        """
        self.spacetime = (self.space, t)


class DataReshape(SpaceTime):
    """Reshape data to create a spacetime matrix."""

    def __init__(self, spacetime=None):
        """
        :param spacetime: spacetime dimensions, defaults to None
        :type spacetime: None, tuple, optional
        """
        super().__init__(spacetime=spacetime)
    
    def variable2matrix(self, variable, dimension):
        """Transform variable to matrix.
        
        :param variable: variable to be transformed
        :param dimension: dimension of :param variable:
            
        :type variable: float, int, list, tuple, numpy.ndarray
        :type dimension: str

        :return: variable as matrix in space-time
        :rtype: numpy.ndarray
        """
        # # input check
        # dimension-type
        dimensions = ('space', 'time')
        if dimension not in dimensions:
            msg = f'{dimension} not in {dimensions}.'
            raise ValueError(msg)

        # dimension-value
        variable = self.variable2array(variable)
        self.dimension_value(variable, dimension)

        # # transformation
        if dimension == 'space':
            return np.tile(variable, (self.time, 1)).transpose()
        elif dimension == 'time':
            return np.tile(variable, (self.space, 1))

    def dimension_value(self, variable, dimension):
        """Check consistency between variable's dimensions and the defined spacetime dimensions.

        :param variable: variable to be checked
        :param dimension: dimension under consideration

        :type variable: list, tuple, numpy.ndarray
        :type dimension: str
        """
        try:
            _ = len(variable)
        except TypeError:
            variable = [variable]

        if not len(variable) == getattr(self, dimension):
            msg = f'Incorrect variable size, {len(variable)} =/= {getattr(self, dimension)}.'
            raise ValueError(msg)

    @staticmethod
    def variable2array(variable):
        """"Transform variable to numpy.array (if float or string).
        
        :param variable: variable to be transformed
        :type variable: float, int, list, numpy.ndarray

        :return: variable as array
        :rtype: numpy.ndarray
        """
        if isinstance(variable, str):
            msg = f'Variable cannot be of {type(variable)}.'
            raise NotImplementedError(msg)
        elif isinstance(variable, (float, int)):
            return np.array([float(variable)])
        elif isinstance(variable, (list, tuple)):
            return np.array(variable)
        elif isinstance(variable, np.ndarray) and not variable.shape:
            return np.array([variable])
        return variable

    def matrix2array(self, matrix, dimension, conversion=None):
        """Transform matrix to array.

        :param matrix: variable as matrix in spacetime
        :param dimension: dimension to convert matrix to
        :param conversion: how to convert the matrix to an array, defaults to None
            None    :   take the last value
            'mean'  :   take the mean value
            'max'   :   take the maximum value
            'min'   :   take the minimum value
            'sum'   :   take the summation

        :type matrix: numpy.ndarray
        :type dimension: str
        :type conversion: None, str, optional

        :return: variable as array
        :rtype: numpy.ndarray
        """
        # # input check
        # dimension-type
        dimensions = ('space', 'time')
        if dimension not in dimensions:
            msg = f'{dimension} not in {dimensions}.'
            raise ValueError(msg)

        # input as numpy.array
        matrix = np.array(matrix)

        # dimension-value
        if not matrix.shape == self.spacetime:
            if not matrix.shape[:2] == self.spacetime:
                msg = f'Matrix-shape does not correspond with spacetime-dimensions:' \
                      f'\n{matrix.shape} =/= {self.spacetime}'
                raise ValueError(msg)

        # conversion-strategy
        conversions = (None, 'mean', 'max', 'min', 'sum')
        if conversion not in conversions:
            msg = f'{conversion} not in {conversions}.'
            raise ValueError(msg)

        # # transformation
        # last position
        if conversion is None:
            if dimension == 'space':
                return matrix[:, -1]
            elif dimension == 'time':
                return matrix[-1, :]

        # conversion
        if dimension == 'space':
            return getattr(matrix, conversion)(axis=1)
        elif dimension == 'time':
            return getattr(matrix, conversion)(axis=0)
    

class Processes:
    """Processes included in CoralModel simulations."""
    
    def __init__(self, fme=True, tme=True, pfd=True):
        """
        :param fme: flow micro-environment, defaults to True
        :param tme: thermal micro-environment, defaults to True
        :param pfd: photosynthetic flow dependency, defaults to True

        :type fme: bool, optional
        :type tme: bool, optional
        :type pfd: bool, optional
        """
        self.pfd = pfd

        if not pfd:
            if fme:
                print(
                    f'WARNING: Flow micro-environment (FME) not possible '
                    f'when photosynthetic flow dependency (PFD) is disabled.'
                )
            self.fme = False
            self.tme = False

        else:
            self.fme = fme
            if not fme:
                if tme:
                    print(
                        f'WARNING: Thermal micro-environment (TME) not possible '
                        f'when flow micro-environment is disabled.'
                    )
                self.tme = False

            else:
                self.tme = tme

        if tme:
            print('WARNING: Thermal micro-environment not fully implemented yet.')

        if not pfd:
            print('WARNING: Exclusion of photosynthetic flow dependency not fully implemented yet.')
            
            
class Constants:
    """Object containing all constants used in CoralModel simulations."""
    
    def __init__(self, processes, lac_default=None, light_spreading_max=None,
                 turbulence_coef=None, inertia_coef=None,
                 friction_coef=None, kin_viscosity=None, therm_diff=None, spacing_ratio=None, wc_angle=None, rd=None,
                 theta=None, err=None, maxiter_k=None, maxiter_aw=None, thermal_coef=None, absorptivity=None,
                 therm_cond=None, pa_rate=None, sat_intensity_max=None, photo_max=None,
                 beta_sat_intensity=None, beta_photo=None, act_energy=None, gas_constant=None, thermal_variability=None,
                 nn=None, pfd_min=None, ucr=None, r_growth=None, r_recovery=None, r_mortality=None, r_bleaching=None,
                 calcification_const=None, arg_sat_default=None, omega0=None, kappa0=None, prop_form=None,
                 prop_plate=None, prop_plate_flow=None, prop_space=None, prop_space_light=None, prop_space_flow=None,
                 u0=None, rho_c=None, sigma_tensile=None, drag_coef=None, rho_w=None, no_larvae=None,
                 prob_settle=None, d_larvae=None):
        """
        Parameters
        ----------
        processes : Processes
            Definition of the processes that are included, specified by means
            of the Processes-object.
        
        > light micro-environment
        lac_default : numeric, optional
            Constant light-attenuation coefficient; is used when no time-series
            is available [m-1]. The default is 0.1.
        light_spreading_max : numeric, optional
            Maximum spreading of light as measured at the water-air interface
            [rad]. The default is 0.5*pi.
        
        > flow micro-environment
        turbulence_coef : float, optional
            Smagorinsky coefficient [-]. The default is 0.17.
        inertia_coef : float, optional
            Inertia coefficient [-]. The default is 1.7.
        friction_coef : float, optional
            Friction coefficient [-]. The default is 0.01.
        kin_viscosity : float, optional
            Kinematic viscosity of water [m2 s-1]. The default is 1e6.
        therm_diff : float, optional
            Thermal diffusivity of water [m2 s-1]. The default is 1e-7.
        spacing_ratio : float, optional
            Ratio of lateral over streamwise spacing of corals [-]. The default
            is 2.
        wc_angle : float, optional
            Angle between current- and wave-induced flows [rad]. The default
            is 0.
        rd : float, optional
            Velocity boundary layer wall-coordinate [-]. The default is 500.
        theta :  float, optional
            Update ratio for above-canopy flow [-]. The default is 0.5.
        err :  float, optional
            Maximum allowed relative error [-]. The default is 1e-6.
        maxiter_k :  float, optional
            Maximum number of iterations taken over the canopy layers. The
            default is 1e5.
        maxiter_aw :  float, optional
            Maximum number of iterations to solve the complex-valued wave-
            attenuation coefficient. The default is 1e5.
        
        > thermal micro-environment
        thermal_coef : float, optional
            Morphological thermal coefficient [-]. The default is 80.
        absorptivity : float, optional
            Absorptivity of coral [-]. The default is 0.4.
        therm_cond : float, optional
            Thermal conductivity [J m-1 s-1 K-1]. The default is 0.6089.
        
        > photosynthetic light dependency
        pa_rate : float, optional
            Photo-acclimation rate [d-1]. The default is 0.6.
        sat_intensity_max : float, optional
            Maximum value of the quasi steady-state for the saturation light-
            intensity [umol photons m-2 s-1]. The default is 372.32.
        photo_max : float, optional
            Maximum value of the quasi steady-state for the maximum
            photosynthetic efficiency [-]. The default is 1.
        beta_sat_intensity : float, optional
            Exponent of the quasi steady-state for the saturation light-
            intensity [-]. The default is 0.34.
        beta_photo : float, optional
            Exponent of the quasi steady-state for the maximum photosynthetic
            efficiency [-]. The default is 0.09.
        
        > photosynthetic thermal dependency
        act_energy : float, optional
            Activation energy [J mol-1]. The default is 6e4.
        gas_constant : float, optional
            Gas constant [J K-1 mol-1]. The default is 8.31446261815324.
        thermal_variability : float, optional
            Thermal-acclimation coefficient [-]. The default is 2.45.
        nn : float, optional
            Thermal-acclimation period [yrs]. The default is 60.
        
        > photosynthetic flow dependency
        pfd_min : float, optional
            Minimum photosynthetic flow dependency [-]. The default is
            0.68886964.
        ucr : float, optional
            Minimum flow velocity at which photosynthesis is not limited by
            flow [m s-1]. The default is (1) 0.17162374 if flow micro-
            environment is enabled; and (2) 0.5173... if flow micro-environment
            is disabled.
        
        > population states
        r_growth : float, optional
            Growth rate [d-1]. The default is 0.002.
        r_recovery : float, optional
            Recovering rate [d-1]. The default is 0.2.
        r_mortality : float, optional
            Mortality rate [d-1]. The default is 0.04.
        r_bleaching : float, optional
            Bleaching rate [d-1]. The default is 8.
        
        > calcification
        calcification_const : float, optional
            Calcification constant [kg m-2 d-1].. The default is 0.5.
        arg_sat_default : float, optional
            Constant aragonite saturation state (is used when no time-series of
            the parameter is available) [-]. The default is 5.
        omega0 : float, optional
            Aragonite dissolution state [-]. The default is 0.14587415.
        kappa0 : float, optional
            Modified Michaelis-Menten half-rate coefficient [-]. The default
            is 0.66236107.
        
        > morphological development
        prop_form : float, optional
            Overall form proportionality constant [-]. The default is 0.1.
        prop_plate : float, optional
            Overall plate proportionality constant [-]. The default is 0.5.
        prop_plate_flow : float, optional
            Flow plate proportionality constant [-]. The default is 0.1.
        prop_space : float, optional
            Overall spacing proportionality constant [-]. The default is
            0.5 / sqrt(2).
        prop_space_light : float, optional
            Light spacing proportionality constant [-]. The default is 0.1.
        prop_space_flow : float, optional
            Flow spacing proportionality constant [-]. The default is 0.1.
        u0 : float, optional
            Base-line flow velocity [m s-1]. The default is 0.2.
        rho_c : float, optional
            Density of coral [kg m-3]. The default is 1600.
        
        > dislodgement criterion
        sigma_tensile : float, optional
            Tensile strength of substratum [N m-2]. The default is 2e5.
        drag_coef : float, optional
            Drag coefficient [-]. The default is 1.
        rho_w : float, optional
            Density of water [kg m-3]. The default is 1025.

        > coral recruitment
        no_larvae : float, optional
            Number of larvae released during mass spawning event [-]. The
            default is 1e6.
        prob_settle : float, optional
            Probability of settlement [-]. The default is 1e-4.
        d_larvae : float, optional
            Larval diameter [m]. The default is 1e-3.

        """
        def default(x, default_value):
            """Set default value if no custom value is provided."""
            if x is None:
                return default_value
            return x
        
        # light micro-environment
        self.Kd0 = default(lac_default, .1)
        self.theta_max = default(light_spreading_max, .5 * np.pi)

        # flow mirco-environment
        self.Cs = default(turbulence_coef, .17)
        self.Cm = default(inertia_coef, 1.7)
        self.Cf = default(friction_coef, .01)
        self.nu = default(kin_viscosity, 1e-6)
        self.alpha = default(therm_diff, 1e-7)
        self.psi = default(spacing_ratio, 2)
        self.wcAngle = default(wc_angle, 0.)
        self.rd = default(rd, 500)
        self.numericTheta = default(theta, .5)
        self.err = default(err, 1e-3)
        self.maxiter_k = int(default(maxiter_k, 1e5))
        self.maxiter_aw = int(default(maxiter_aw, 1e5))

        # thermal micro-environment
        self.K0 = default(thermal_coef, 80.)
        self.ap = default(absorptivity, .4)
        self.k = default(therm_cond, .6089)

        # photosynthetic light dependency
        self.iota = default(pa_rate, .6)
        self.ik_max = default(sat_intensity_max, 372.32)
        self.pm_max = default(photo_max, 1.)
        self.betaI = default(beta_sat_intensity, .34)
        self.betaP = default(beta_photo, .09)

        # photosynthetic thermal dependency
        self.Ea = default(act_energy, 6e4)
        self.R = default(gas_constant, 8.31446261815324)
        self.k_var = default(thermal_variability, 2.45)
        self.nn = default(nn, 60)

        # photosynthetic flow dependency
        self.pfd_min = default(pfd_min, .68886964)
        self.ucr = default(ucr, .17162374 if processes.fme else .5173)

        # population dynamics
        self.r_growth = default(r_growth, .002)
        self.r_recovery = default(r_recovery, .2)
        self.r_mortality = default(r_mortality, .04)
        self.r_bleaching = default(r_bleaching, 8.)

        # calcification
        self.gC = default(calcification_const, .5)
        self.omegaA0 = default(arg_sat_default, 5.)
        self.omega0 = default(omega0, .14587415)
        self.kappaA = default(kappa0, .66236107)

        # morphological development
        self.prop_form = default(prop_form, .1)
        self.prop_plate = default(prop_plate, .5)
        self.prop_plate_flow = default(prop_plate_flow, .1)
        self.prop_space = default(prop_space, .5 / np.sqrt(2.))
        self.prop_space_light = default(prop_space_light, .1)
        self.prop_space_flow = default(prop_space_flow, .1)
        self.u0 = default(u0, .2)
        self.rho_c = default(rho_c, 1600.)

        # dislodgement criterion
        self.sigma_t = default(sigma_tensile, 2e5)
        self.Cd = default(drag_coef, 1.)
        self.rho_w = default(rho_w, 1025.)

        # coral recruitment
        self.no_larvae = default(no_larvae, 1e6)
        self.prob_settle = default(prob_settle, 1e-4)
        self.d_larvae = default(d_larvae, 1e-3)


def coral_only_function(coral, function, args, no_cover_value=0):
    """Only execute the function when there is coral cover.

    :param coral: coral object
    :param function: function to be executed
    :param args: input arguments of the function
    :param no_cover_value: default value in absence of coral cover

    :type coral: Coral
    :type args: tuple
    :type no_cover_value: float, optional
    """
    try:
        size = len(coral.cover)
    except TypeError:
        size = 1

    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, (float, int)) or (isinstance(arg, np.ndarray) and not arg.shape):
            args[i] = np.repeat(arg, size)
        elif not len(arg) == size:
            msg = f'Sizes do not match up, {len(arg)} =/= {size}.'
            raise ValueError(msg)

    output = no_cover_value * np.ones(size)
    output[coral.cover > 0] = function(*[
        arg[coral.cover > 0] for arg in args
    ])
    return output
