"""
coral_mostoel - environment

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""

import pandas as pd
import numpy as np
import distutils.util as du
import os

class Constants:
    """Object containing all constants used in coral_model simulations."""

    def __init__(self):
        
        """
                
        
        # Key = value           ! Default   ! Definition [units]
        #--------------------------------------------------------------------------------------------------------------------
        # processes
        fme = False             # False     ! flow micro-environment
        tme = False             # False     ! thermal micro-environment
        pfd = False             # False     ! ??
        warn_proc = True        # True      ! print warning for incompatible processes
        
        # light attenuation
        Kd0 = 0.1               #  .1       ! constant light-attenuation coefficient [m-1]; used when no time-series provided
        theta_max = 0.5         #  0.5      ! maximum spreading of light [rad]; defined at water-air interface
        
        # flow mirco-environment
        Cs = 0.17               #  .17      ! Smagorinsky coefficient [-]
        Cm = 1.7                #  1.7      ! inertia coefficient [-]
        Cf = 0.01               #  .01      ! friction coefficient [-]
        nu = 1e-6               #  1e-6     ! kinematic viscosity of water [m2 s-1]
        alpha = 1e-7            #  1e-7     ! thermal diffusivity of water [m2 s-1]
        psi = 2                 #  2        ! ratio of lateral over longitudinal spacing of corals [-]
        wcAngle = 0.            #  0.       ! angle between current- and wave-induced flows [rad]
        rd = 500                #  500      ! velocity boundary layer wall-coordinate [-]
        numericTheta = 0.5      #  .5       ! update ratio for above-canopy flow [-]
        err = 1e-3              #  1e-3     ! maximum allowed relative error for drag coefficient estimation [-]
        maxiter_k = 1e5         #  1e5      ! maximum number of iterations taken over canopy layers
        maxiter_aw = 1e5        #  1e5      ! maximum number of iterations to solve complex-valued wave-attenuation coefficient
        
        # thermal micro-environment
        K0 = 80.                #  80.      ! morphological thermal coefficient [-]
        ap = 0.4                #  .4       ! absorptivity of coral [-]
        k = 0.6089              #  .6089    ! thermal conductivity [K m-1 s-1 K-1]
        
        # photosynthetic light dependency
        iota = .6               # .6        ! photo-acclimation rate [d-1]
        ik_max = 372.32         # 372.32    ! maximum quasi steady-state saturation light-intensity [umol photons m-2 s-1]
        pm_max = 1.             #  1.       ! maximum quasi steady-state maximum photosynthetic efficiency [-]
        betaI = .34             # .34       ! exponent of the quasi steady-state saturation light-intensity [-]
        betaP = .09             # .09       ! exponent of the quasi steady-state maximum photosynthetic efficiency [-]
        
        # photosynthetic thermal dependency
        Ea = 6e4                # 6e4       ! activation energy [J mol-1]
        R = 8.31446261815324    # 8.31446261815324 ! gas constant [J K-1 mol-1]
        k_var = 2.45            # 2.45      ! thermal-acclimation coefficient [-]
        nn = 60                 # 60        ! thermal-acclimation period [y]
        
        # photosynthetic flow dependency
        pfd_min = .68886964     # .68886964 ! minimum photosynthetic flow dependency [-]
        ucr = .5173             #.17162374 if processes.fme else .5173) ! minimum flow velocity at which photosynthesis is not limited by flow [m s-1]
        
        
        # population dynamics
        r_growth = .002         # 0.002     ! growth rate [d-1]
        r_recovery = .2         # .2        ! recovery rate [d-1]
        r_mortality = .04       # .04       ! mortality rate [d-1]
        r_bleaching =  8.       # 8.        ! bleaching rate [d-1]
        
        # calcification
        gC = .5                 # .5        ! calcification constant [kg m-2 d-1]
        omegaA0 = 5.            # 5         ! aragonite saturation state used in absence of time-series [-]
        omega0 = .14587415      # .14587415 ! aragonite dissolution state [-]
        kappaA = .66236107      # .66236107 ! modified Michaelis-Menten half-rate coefficient [-]
        #
        # morphological development
        prop_form = .1          # .1        ! overall form proportionality constant [-]
        prop_plate = .5         # .5        ! overall plate proportionality constant [-
        prop_plate_flow = .1    # .1        !  flow plate proportionality constant [-]
        prop_space = .5         # .5/np.sqrt(2.) ! overall space proportionality constant [-]
        prop_space_light = .1   # .1      ! light space proportionality constant [-]
        prop_space_flow = .1    # .1        ! flow space proportionality constant [-]
        u0 = .2                 # .2        ! base-line flow velocity [m s-1]
        rho_c = 1600.           # 1600.     ! density of coral [kg m-3]
        #
        # dislodgement criterion
        sigma_t = 2e5           # 2e5       ! tensile strength of substratum [N m-2]
        Cd = 1.                 # 1.        ! drag coefficient [-]
        rho_w = 1025.           # 1025.     ! density of water [kg m-3]
        #
        # coral recruitment
        no_larvae = 1e6         # 1e6       ! number of larvae released during mass spawning event [-]
        prob_settle = 1e-4      # 1e-4      ! probability of settlement [-]
        d_larvae = 1e-3         # 1e-3      ! larval diameter [m]

        """
                # Processes
        self.fme = None
        self.tme = None
        self.pfd = None
        self.warn_proc = None
        
        # light micro-environment
        self.Kd0 = None
        self.theta_max = None

        # flow micro-environment
        self.Cs =  None
        self.Cm =  None
        self.Cf = None
        self.nu =  None
        self.alpha =  None
        self.psi =  None
        self.wcAngle =  None
        self.rd =  None
        self.numericTheta =  None
        self.err =  None
        self.maxiter_k =  None
        self.maxiter_aw =  None

        # thermal micro-environment
        self.K0 =  None
        self.ap =  None
        self.k =  None

        # photosynthetic light dependency
        self.iota =  None
        self.ik_max =  None
        self.pm_max =  None
        self.betaI = None
        self.betaP =  None

        # photosynthetic thermal dependency
        self.Ea = None
        self.R = None
        self.k_var =  None
        self.nn =  None

        # photosynthetic flow dependency
        self.pfd_min =  None
        self.ucr = None

        # population dynamics
        self.r_growth =  None
        self.r_recovery =  None
        self.r_mortality = None
        self.r_bleaching = None

        # calcification
        self.gC =  None
        self.omegaA0 =  None
        self.omega0 = None
        self.kappaA = None

        # morphological development
        self.prop_form =  None
        self.prop_plate =  None
        self.prop_plate_flow =  None
        self.prop_space =  None
        self.prop_space_light = None
        self.prop_space_flow =  None
        self.u0 = None
        self.rho_c = None

        # dislodgement criterion
        self.sigma_t = None
        self.Cd = None
        self.rho_w = None

        # coral recruitment
        self.no_larvae =  None
        self.prob_settle = None
        self.d_larvae =  None
    

    def check_processes(self):
        if not self.pfd:
            if self.fme and self.warn_proc:
                print(
                    f'WARNING: Flow micro-environment (FME) not possible '
                    f'when photosynthetic flow dependency (PFD) is disabled.'
                )
            self.fme = False
            self.tme = False

        else:
            if not self.fme:
                if self.tme and self.warn_proc:
                    print(
                        f'WARNING: Thermal micro-environment (TME) not possible '
                        f'when flow micro-environment is disabled.'
                    )
                self.tme = False

            else:
                self.tme = self.tme

        if self.tme and self.warn_proc:
            print('WARNING: Thermal micro-environment not fully implemented yet.')

        if not self.pfd and self.warn_proc:
            print('WARNING: Exclusion of photosynthetic flow dependency not fully implemented yet.')


    def read_it(self,inp_file):        
        self.inpfile=inp_file
            
        keyvals={}
        with open(self.inpfile) as f:
            for line in f:
                if(len(line)>1):
                    linee = line
                    if (line.count("#")>0):
                        linee,dum = line.split ("#")
                    if(len(linee)>0):
                        name, value = linee.split("=")
                        value=value.lower().strip()
                        try:
                            keyvals[name.strip()] = float(value)
                        except (ValueError):
                            keyvals[name.strip()]=bool(du.strtobool(value))

        def default(x, default_value):
            """Set default value if no custom value is provided."""
            xx = keyvals.get(x)
            if (xx is None):
                xx = default_value
            return xx

        # Processes
        self.fme = default ("fme",False) 
        self.tme = default ("tme",False)
        self.pfd = default ("pfd",False)
        self.warn_proc = default("warn_proc",True)
        
        # light micro-environment
        self.Kd0 = default("Kd0", .1)
        self.theta_max = default("theta_max", .5) * np.pi

        # flow micro-environment
        self.Cs = default("Cs", .17)
        self.Cm = default("Cm", 1.7)
        self.Cf = default("Cf", .01)
        self.nu = default("nu", 1e-6)
        self.alpha = default("alpha", 1e-7)
        self.psi = default("psi", 2)
        self.wcAngle = default("wcAngle", 0.)
        self.rd = default("rd", 500)
        self.numericTheta = default("numericTheta", .5)
        self.err = default("err", 1e-3)
        self.maxiter_k = int(default("maxiter_k", 1e5))
        self.maxiter_aw = int(default("maxiter_aw", 1e5))

        # thermal micro-environment
        self.K0 = default("K0", 80.)
        self.ap = default("ap", .4)
        self.k = default("k", .6089)

        # photosynthetic light dependency
        self.iota = default("iota", .6)
        self.ik_max = default("ik_max", 372.32)
        self.pm_max = default("pm_max", 1.)
        self.betaI = default("betaI", .34)
        self.betaP = default("beta_P", .09)

        # photosynthetic thermal dependency
        self.Ea = default("Ea", 6e4)
        self.R = default("R", 8.31446261815324)
        self.k_var = default("k_var", 2.45)
        self.nn = default("nn", 60)

        # photosynthetic flow dependency
        self.pfd_min = default("pfd_min", .68886964)
        self.ucr = default("ucr", .5173)

        # population dynamics
        self.r_growth = default("r_growth", .002)
        self.r_recovery = default("r_recovery", .2)
        self.r_mortality = default("r_mortality", .04)
        self.r_bleaching = default("r_bleaching", 8.)

        # calcification
        self.gC = default("gC", .5)
        self.omegaA0 = default("omegaA0", 5.)
        self.omega0 = default("omega0", .14587415)
        self.kappaA = default("kappaA", .66236107)

        # morphological development
        self.prop_form = default("prop_form", .1)
        self.prop_plate = default("prop_plate", .5)
        self.prop_plate_flow = default("prop_plate_flow", .1)
        self.prop_space = default("prop_space", .5) / np.sqrt(2.)
        self.prop_space_light = default("prop_space_light", .1)
        self.prop_space_flow = default("prop_space_flow", .1)
        self.u0 = default("u0", .2)
        self.rho_c = default("rho_c", 1600.)

        # dislodgement criterion
        self.sigma_t = default("sigma_t", 2e5)
        self.Cd = default("Cd", 1.)
        self.rho_w = default("rho_w", 1025.)

        # coral recruitment
        self.no_larvae = default("no_larvae", 1e6)
        self.prob_settle = default("prob_settle", 1e-4)
        self.d_larvae = default("d_larvae", 1e-3)
        
        # check processes for consistency
        self.check_processes



class Environment:
    # TODO: Make this class robust

    _dates = None
    _light = None
    _light_attenuation = None
    _temperature = None
    _aragonite = None
    _storm_category = None

    @property
    def light(self):
        """Light-intensity in micro-mol photons per square metre-second."""
        return self._light

    @property
    def light_attenuation(self):
        """Light-attenuation coefficient in per metre."""
        return self._light_attenuation

    @property
    def temperature(self):
        """Temperature time-series in either Celsius or Kelvin."""
        return self._temperature

    @property
    def aragonite(self):
        """Aragonite saturation state."""
        return self._aragonite

    @property
    def storm_category(self):
        """Storm category time-series."""
        return self._storm_category

    @property
    def temp_kelvin(self):
        """Temperature in Kelvin."""
        if all(self.temperature.values < 100) and self.temperature is not None:
            return self.temperature + 273.15
        return self.temperature

    @property
    def temp_celsius(self):
        """Temperature in Celsius."""
        if all(self.temperature.values > 100) and self.temperature is not None:
            return self.temperature - 273.15
        return self.temperature

    @property
    def temp_mmm(self):
        monthly_mean = self.temp_kelvin.groupby([
            self.temp_kelvin.index.year, self.temp_kelvin.index.month
        ]).agg(['mean'])
        monthly_maximum_mean = monthly_mean.groupby(level=0).agg(['min', 'max'])
        monthly_maximum_mean.columns = monthly_maximum_mean.columns.droplevel([0, 1])
        return monthly_maximum_mean

    @property
    def dates(self):
        """Dates of time-series."""
        if self._dates is not None:
            d = self._dates
        elif self.light is not None:
            # TODO: Check column name of light-file
            d = self.light.reset_index().drop('light', axis=1)
            self._dates=d
        elif self.temperature is not None:
            d = self.temperature.reset_index().drop('sst', axis=1)
            self._dates=d
        else:
            msg = f'No initial data on dates provided.'
            raise ValueError(msg)
        return pd.to_datetime(d['date'])

    def set_dates(self, start_date, end_date):
        """Set dates manually, ignoring possible dates in environmental time-series.

        :param start_date: first date of time-series
        :param end_date: last date of time-series

        :type start_date: str, datetime.date
        :type end_date: str, datetime.date
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        self._dates = pd.DataFrame({'date': dates})

    def set_parameter_values(self, parameter, value, pre_date=None):
        """Set the time-series data to a time-series, or a default value. In case :param value: is not iterable, the
        :param parameter: is assumed to be constant over time. In case :param value: is iterable, make sure its length
        complies with the simulation length.

        Included parameters:
            light                       :   incoming light-intensity [umol photons m-2 s-1]
            LAC / light_attenuation     :   light attenuation coefficient [m-1]
            temperature                 :   sea surface temperature [K]
            aragonite                   :   aragonite saturation state [-]
            storm                       :   storm category, annually [-]

        :param parameter: parameter to be set
        :param value: default value
        :param pre_date: time-series start before simulation dates [yrs]

        :type parameter: str
        :type value: float, list, tuple, numpy.ndarray, pandas.DataFrame
        :type pre_date: None, int, optional
        """

        def set_value(val):
            """Function to set default value."""
            if pre_date is None:
                return pd.DataFrame({parameter: val}, index=self.dates)

            dates = pd.date_range(self.dates.iloc[0] - pd.DateOffset(years=pre_date), self.dates.iloc[-1], freq='D')
            return pd.DataFrame({parameter: val}, index=dates)

        if self._dates is None:
            msg = f'No dates are defined. ' \
                f'Please, first specify the dates before setting the time-series of {parameter}; ' \
                f'or make use of the \"from_file\"-method.'
            raise TypeError(msg)

        if parameter == 'LAC':
            parameter = 'light_attenuation'

        daily_params = ('light', 'light_attenuation', 'temperature', 'aragonite')
        if parameter in daily_params:
            setattr(self, f'_{parameter}', set_value(value))
        elif parameter == 'storm':
            years = set(self.dates.dt.year)
            self._storm_category = pd.DataFrame(data=value, index=years)
        else:
            msg = f'Entered parameter ({parameter}) not included. See documentation.'
            raise ValueError(msg)

    def from_file(self, parameter, file, folder):
        """Read the time-series data from a file.

        Included parameters:
            light                       :   incoming light-intensity [umol photons m-2 s-1]
            LAC / light_attenuation     :   light attenuation coefficient [m-1]
            temperature                 :   sea surface temperature [K]
            aragonite                   :   aragonite saturation state [-]
            storm                       :   storm category, annually [-]

        :param parameter: parameter to be read from file
        :param file: file name, incl. file extension
        :param folder: folder directory, defaults to None

        :type parameter: str
        :type file: str
        :type folder: str
        """
        # TODO: Include functionality to check file's existence
        #  > certain files are necessary: light, temperature

        def read_index(fil):
            """Function applicable to time-series in Pandas."""
            time_series = pd.read_csv(f,sep = '\t')
            time_series['date'] = pd.to_datetime(time_series['date'])
            time_series.set_index('date', inplace=True)
            return time_series

        f = os.path.join(folder,file)

        if parameter == 'LAC':
            parameter = 'light_attenuation'

        daily_params = ('light', 'light_attenuation', 'temperature', 'aragonite')
        if parameter in daily_params:
            setattr(self, f'_{parameter}', read_index(f))
        elif parameter == 'storm':
            self._storm_category = pd.read_csv(f, sep='\t')
            self._storm_category.set_index('year', inplace=True)
        else:
            msg = f'Entered parameter ({parameter}) not included. See documentation.'
            raise ValueError(msg)
