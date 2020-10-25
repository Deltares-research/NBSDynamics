# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:32:23 2020

@author: hendrick
"""

import numpy as np
import pandas as pd
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import bmi.wrapper
import os
from scipy.optimize import newton
from tqdm import tqdm
import datetime
from netCDF4 import Dataset
import faulthandler
faulthandler.enable()

# =============================================================================
# # # # specify directories of ddl- and input-files
# =============================================================================
model_folder = os.path.join('SensitivityAnalysis', 'sa007')

# Delft3D directories
D3D_HOME = os.path.join('p:\\11202744-008-vegetation-modelling', 'code_1709',
                        'windows', 'oss_artifacts_x64_63721', 'x64')
dflow_dir = os.path.join(D3D_HOME, 'dflowfm', 'bin', 'dflowfm.dll')
dimr_path = os.path.join(D3D_HOME, 'dimr', 'bin', 'dimr_dll.dll')
# work directory
workdir = os.path.join('p:\\11202744-008-vegetation-modelling', 'students',
                       'GijsHendrickx', 'models', model_folder)
inputdir = os.path.join(workdir, 'timeseries')
# input files (Delft3D)
config_file = os.path.join(workdir, 'dimr_config.xml')
mdufile = os.path.join(workdir, 'fm', 'FlowFM.mdu')
# print directories and input-files as check
print('Model              : {0}\n'.format(workdir))
print('Delft3D home       : {0}'.format(D3D_HOME))
print('DIMR-directory     : {0}'.format(dimr_path))
print('Configuration file : {0}'.format(config_file))

# =============================================================================
# # # # prepare locations
# =============================================================================
# # print directories of input- and output-files
print('\nTime-series dir.   : {0}'.format(inputdir))

# # intermediate figures
figfolder = os.path.join(workdir, 'figures')
# check existence and create if necessary
if not os.path.exists(figfolder):
    os.mkdir(figfolder)
    print('New folder created : {0}'.format(figfolder))
print('Figure directory   : {0}'.format(figfolder))

# # output files
outputfolder = os.path.join(workdir, 'output')
# check existance and create if necessary
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
    print('New folder created : {0}'.format(outputfolder))
print('Output directory   : {0}'.format(outputfolder))

# =============================================================================
# # # # environmental conditions - files
# =============================================================================
fLight = 'TS_PAR.txt'
fLAC = 'TS_LAC.txt'
fTemperature = 'TS_SST.txt'
fAcidity = 'TS_ARG.txt'
fStorm = 'TS_stormcat.txt'

# =============================================================================
# # # # create correct environment
# =============================================================================
os.environ['PATH'] = (os.path.join(D3D_HOME, 'share', 'bin') + ';' +
                      os.path.join(D3D_HOME, 'dflowfm', 'bin') + ';' +
                      os.path.join(D3D_HOME, 'dimr', 'bin') + ';' +
                      os.path.join(D3D_HOME, 'dwaves', 'bin') + ';' +
                      os.path.join(D3D_HOME, 'esmf', 'scripts') + ';' +
                      os.path.join(D3D_HOME, 'swan', 'scripts'))
# print created environment as check
print('\nEnvironment        : {0}\n'
      .format(os.path.join(D3D_HOME, 'share', 'bin')) +
      '                     {0}\n'
      .format(os.path.join(D3D_HOME, 'dflowfm', 'bin')) +
      '                     {0}\n'
      .format(os.path.join(D3D_HOME, 'dimr', 'bin')) +
      '                     {0}\n'
      .format(os.path.join(D3D_HOME, 'dwaves', 'bin')) +
      '                     {0}\n'
      .format(os.path.join(D3D_HOME, 'esmf', 'scripts')) +
      '                     {0}\n'
      .format(os.path.join(D3D_HOME, 'swan', 'scripts')))

# =============================================================================
# # # # define and initialize wrappers
# =============================================================================
# define DFM wrapper
modelFM = bmi.wrapper.BMIWrapper(engine=dflow_dir, configfile=mdufile)
# define DIMR wrapper
modelDIMR = bmi.wrapper.BMIWrapper(engine=dimr_path, configfile=config_file)
# initialise model
modelDIMR.initialize()

print('Model initialized.\n')

# =============================================================================
# # # # set the pointers to important model variables of FlowFM
# =============================================================================
# number of boxes, including boundary boxes
ndx = modelFM.get_var('ndx')
# number of non-boundary boxes, i.e. within-domain boxes
ndxi = modelFM.get_var('ndxi')
# x-coord. of the center of gravity of the boxes
xzw = modelFM.get_var('xzw')
# y-coord. of the center of gravity of the boxes
yzw = modelFM.get_var('yzw')
# total number of links between boxes
lnx = modelFM.get_var('lnx')
# number of links between within-domain boxes
lnxi = modelFM.get_var('lnxi')
# link martrix between adjacent boxes [ln, 2] matrix
ln = modelFM.get_var('ln')
# distance between the centers of adjacent boxes
dx = modelFM.get_var('dx')
# width of the interface between adjacent boxes
wu = modelFM.get_var('wu')
# surface area of the boxes
ba = modelFM.get_var('ba')


# =============================================================================
# # # # set time parameters for coupled model
# =============================================================================
# # time-span
# start year
Ystart = 2000
# simulation time [yrs]
Y = 100
# year range
years = np.arange(Ystart, Ystart + Y)

# # model time per vegetation step [s]
mtpervt = 43200
# storm
mtpervt_storm = 86400

# =============================================================================
# # # # define output
# =============================================================================
# # # map > full spatial extent
# # data to output file
# representative light-intensity > { Iz }
L2mfile = True
# mean flow > { uc }
U2mfile = True
# mean coral temperature > { Tc, Tlo, Thi }
T2mfile = True
# mean photosynthesis > { PS }
PS2mfile = True
# population states > { P } > { PH, PR, PP, PB }
P2mfile = True
# calcification > { G }
G2mfile = True
# morphology > { Lc } > { dc, hc, bc, tc, ac, Vc }
M2mfile = True

# # map-file
# map-file directory
mapfile = 'CoralModel_map.nc'
mapfilef = os.path.join(outputfolder, mapfile)
# time-interval > annually

# # # history > time-series
# # data to output file
# light > { Iz }
L2hfile = True
# flow > { uc }
U2hfile = True
# temperature > { Tc, Tlo, Thi }
T2hfile = True
# photosynthesis > { PS }
PS2hfile = True
# population states > { P } > { PH, PR, PP, PB }
P2hfile = True
# calcification > { G }
G2hfile = True
# morphology > { Lc } > { dc, hc, bc, tc, ac, Vc }
M2hfile = True

# # his-file
# location(s)
xynfilef = os.path.join(workdir, 'fm', 'FlowFm_obs.xyn')
xyn = pd.read_csv(xynfilef, header=None, delim_whitespace=True)
xyn.columns = ['x', 'y', 'name']
# his-file directory
hisfile = 'CoralModel_his.nc'
hisfilef = os.path.join(outputfolder, hisfile)
# time-interval > daily

# =============================================================================
# # # # vegetation boundaries
# =============================================================================
xbndmin = min(xzw[range(ndxi)])
xbndmax = max(xzw[range(ndxi)])
ybndmin = min(yzw[range(ndxi)])
ybndmax = max(yzw[range(ndxi)])

xvbndmin = xbndmin
xvbndmax = xbndmax
yvbndmin = ybndmin
yvbndmax = 700.

# =============================================================================
# # # # initial conditions
# =============================================================================
# # initial morphology
dc0 = .05  # m
hc0 = .3  # m
bc0 = .5 * dc0
tc0 = .5 * hc0
ac0 = .2  # m

# # carrying capacity
K = np.zeros(ndxi)
K[np.logical_and.reduce((xzw >= xvbndmin,
                         xzw <= xvbndmax,
                         yzw >= yvbndmin,
                         yzw <= yvbndmax))] = 1.

# # coral cover
P0 = np.array([
        K,
        np.zeros(K.shape),
        np.zeros(K.shape),
        np.zeros(K.shape)
    ]).transpose()

print('Constants set.\n')

# =============================================================================
# # # # intermediate plotting function
# =============================================================================
# triangulate face coordinates for plotting
face_triang = tri.Triangulation(xzw[range(ndxi)], yzw[range(ndxi)])
# define basic plotting routine for model output


def showfld(fld, face_triang, lvls, ttl,
            show=True, save=False, unit=None):
    """
    Function to visualise the intermediate results from the model computations
    in which the grid structure used in Delft3D-FM is translated such that a
    field is presented.
    """
    f = plt.figure()
    plt.tricontourf(face_triang, fld, levels=lvls)
    plt.title(ttl)
    cbar = plt.colorbar(shrink=.5, extend='both', aspect=10)
    if unit is not None:
        cbar.set_label(unit, rotation=270, va='bottom')
    if show:
        plt.show()
    if save:
        f.savefig(os.path.join(figfolder, ('{0}.png'.format(ttl))),
                  bbox_inches='tight')


# =============================================================================
# # # # input modification
# =============================================================================

global spacetime

class DataReshape():
    
    def __init__(self, spacetime):
        """
        Reshape data to matrix with shape [space x time].

        Parameters
        ----------
        spacetime: numeric
            Definition of with dimensions of spacetime: [space, time].

        Returns
        -------
        2D-matrix.

        """
        self.space = int(spacetime[0])
        self.time = int(spacetime[1])
    
    def param2matrix(self, x, dimension):
        """Transform parameter to 2D-matrix."""
        # # dimensions
        dims = ['space', 'time']
        if dimension not in dims:
            raise ValueError(
                'Dimension not in {}'.format(dims))
        # # transformation
        x = self.param2array(x)
        if dimension == 'space':
            M = np.tile(x, (self.time, 1)).transpose()
        elif dimension =='time':
            M = np.tile(x, (self.space, 1))
        return M
    
    @staticmethod
    def param2array(x):
        """Transform parameter to array if float or integer."""
        if isinstance(x, (float, int)):
            x = np.array([float(x)])
        return x

# =============================================================================
# # # # specification of processes included
# =============================================================================

global processes

class Processes():
    
    def __init__(self, fme=True, tme=True, pfd=True):
        """
        Processes included in CoralModel simulations.

        Parameters
        ----------
        fme : bool, optional
            Flow micro-environment. The default is True.
        tme : bool, optional
            Thermal micro-environment. The default is True.
        pfd : bool, optional
            Photosynthetic flow dependency. The default is True.

        Returns
        -------
        Object describing included processes.

        """
        self.pfd = pfd
        if not pfd:
            self.fme = False
            self.tme = False
            if fme:
                print('WARNING: Flow micro-environment (FME) not possible '
                      'when photosynthetic flow dependency (PFD) is disabled.')
        else:
            self.fme = fme
            if not fme:
                self.tme = False
                if tme:
                    print('WARNING: Thermal micro-environment (TME) not '
                          'possible when flow micro-environment is disabled.')
            else:
                self.tme = tme

        if tme:
            print('WARNING: Thermal micro-environment not fully implemented '
                  'yet.')
        if not pfd:
            print('WARNING: Exclusion of photosynthetic flow dependency not '
                  'fully implemented yet.')

processes = Processes(fme=True, tme=False, pfd=True)

# =============================================================================
# # # # definition of constants
# =============================================================================

global constants

class Constants():
    
    def __init__(self, Kd0=None, thetamax=None, Cs=None, Cm=None, Cf=None,
                 nu=None, alpha=None, psi=None, wcAngle=None, rd=None,
                 numericTheta=None, err=None, maxiter_k=None, maxiter_aw=None,
                 K0=None, ap=None, k=None, iota=None, Ikmax=None, Pmmax=None,
                 betaI=None, betaP=None, Ea=None, R=None, Kvar=None, nn=None,
                 Pumin=None, ucr=None, rG=None, rR=None, rM=None, rB=None,
                 gC=None, omegaA0=None, omega0=None, kappaA=None, Xf=None,
                 Xp=None, Xpu=None, Xs=None, XsI=None, Xsu=None, u0=None,
                 rhoc=None, sigmat=None, Cd=None, rhow=None, NoLarvae=None,
                 probSett=None, dLarv=None):
        """
        Object containing all constants used in CoralModel. None indicates the
        use of the default value.

        Parameters
        ----------
        > light micro-environment
        Kd0 : float, optional
            Constant light-attenuation coefficient (is used when no time-series
            of the coefficient are available) [m-1]. The default is 0.1.
        thetamax : float, optional
            Maximum spreading of light as measured at water-air interface
            [rad]. The default is 0.5*pi.
        
        > flow micro-environment
        Cs : float, optional
            Smagorinsky coefficient [-]. The default is 0.17.
        Cm : float, optional
            Inertia coefficient [-]. The default is 1.7.
        Cf : float, optional
            Fricition coefficient [-]. The default is 0.01.
        nu : float, optional
            Kinematic viscosity of water [m2 s-1]. The default is 1e6.
        alpha : float, optional
            Thermal diffusivity of water [m2 s-1]. The default is 1e-7.
        psi : float, optional
            Ratio of lateral over streamwise spacing of corals [-]. The default
            is 2.
        wcAngle : float, optional
            Angle between current- and wave-induced flows [rad]. The default
            is 0.
        rd : float, optional
            Velocity boundary layer wall-coordinate [-]. The default is 500.
        numericTheta :  float, optional
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
        K0 : float, optional
            Morphological thermal coefficient [-]. The default is 80.
        ap : float, optional
            Absorptibity of coral [-]. The default is 0.4.
        k : float, optional
            Thermal conductivity [J m-1 s-1 K-1]. The default is 0.6089.
        
        > photosynthetic light dependency
        iota : float, optional
            Photo-acclimation rate [d-1]. The default is 0.6.
        Ikmax : float, optional
            Maximum value of the quasi steady-state for the saturation light-
            intensity [umol photons m-2 s-1]. The default is 372.32.
        Pmmax : float, optional
            Maximum value of the quasi steady-state for the maximum
            photosynthetic efficiency [-]. The default is 1.
        betaI : float, optional
            Exponent of the quasi steady-state for the saturation light-
            intensity [-]. The default is 0.34.
        betaP : float, optional
            Exponent of the quasi steady-state for the maximum photosynthetic
            efficiency [-]. The default is 0.09.
        
        > photosynthetic thermal dependency
        Ea : float, optional
            Activation energy [J mol-1]. The default is 6e4.
        R : float, optional
            Gas constant [J K-1 mol-1]. The default is 8.31446261815324.
        Kvar : float, optional
            Thermal-acclimation coefficient [-]. The default is 2.45.
        nn : float, optional
            Theraml-acclimation period [yrs]. The default is 60.
        
        > photosynthetic flow dependency
        Pumin : float, optional
            Minimum photosynthetic flow dependency [-]. The default is
            0.68886964.
        ucr : float, optional
            Minimum flow velocity at which photosynthesis is not limited by
            flow [m s-1]. The default is (1) 0.17162374 if flow micro-
            environment is enabled; and (2) 0.5173... if flow micro-environment
            is disabled.
        
        > population states
        rG : float, optional
            Growth rate [d-1]. The default is 0.002.
        rR : float, optional
            Recovering rate [d-1]. The default is 0.2.
        rM : float, optional
            Mortality rate [d-1]. The default is 0.04.
        rB : float, optional
            Bleaching rate [d-1]. The default is 8.
        
        > calcification
        gC : float, optional
            Calcification constant [kg m-2 d-1].. The default is 0.5.
        omegaA0 : float, optional
            Constant aragonite saturation state (is used when no time-series of
            the parameter is available) [-]. The default is 5.
        omega0 : float, optional
            Aragonite dissolution state [-]. The default is 0.14587415.
        kappaA : float, optional
            Modified Michaelis-Menten half-rate coefficient [-]. The default
            is 0.66236107.
        
        > morphological development
        Xf : float, optional
            Overall form proportionality constant [-]. The default is 0.1.
        Xp : float, optional
            Overall plate proportionality constant [-]. The default is 0.5.
        Xpu : float, optional
            Flow plate proportionality constant [-]. The default is 0.1.
        Xs : float, optional
            Overall spacing proportionality constant [-]. The default is
            0.5 / sqrt(2).
        XsI : float, optional
            Light spacing proportionality constant [-]. The default is 0.1.
        Xsu : float, optional
            Flow spacing proportionality constant [-]. The default is 0.1.
        u0 : float, optional
            Base-line flow velocity [m s-1]. The default is 0.2.
        rhoc : float, optional
            Density of coral [kg m-3]. The default is 1600.
        
        > dislodgement criterion
        sigmat : float, optional
            Tensile strength of substratum [N m-2]. The default is 2e5.
        Cd : float, optional
            Drag coefficient [-]. The default is 1.
        rhow : float, optional
            Density of water [kg m-3]. The default is 1025.

        > coral recruitment
        NoLarvae : float, optional
            Number of larvae released during mass spawning event [-]. The
            default is 1e6.
        probSett : float, optional
            Probability of settlement [-]. The default is 1e-4.
        dLarv : float, optional
            Larval diameter [m]. The default is 1e-3.

        """
        def Default(x, default):
            if x is None:
                x = default
            return x
        
        # light micro-environment
        self.Kd0 = Default(Kd0, .1)
        self.thetamax = Default(thetamax, .5*np.pi)
        # flow mirco-environment
        self.Cs = Default(Cs, .17)
        self.Cm = Default(Cm, 1.7)
        self.Cf = Default(Cf, .01)
        self.nu = Default(nu, 1e-6)
        self.alpha = Default(alpha, 1e-7)
        self.psi = Default(psi, 2)
        self.wcAngle = Default(wcAngle, 0.)
        self.rd = Default(rd, 500)
        self.numericTheta = Default(numericTheta, .5)
        self.err = Default(err, 1e-3)
        self.maxiter_k = int(Default(maxiter_k, 1e5))
        self.maxiter_aw = int(Default(maxiter_aw, 1e5))
        # thermal micro-environment
        self.K0 = Default(K0, 80.)
        self.ap = Default(ap, .4)
        self.k = Default(k, .6089)
        # photosynthetic light dependency
        self.iota = Default(iota, .6)
        self.Ikmax = Default(Ikmax, 372.32)
        self.Pmmax = Default(Pmmax, 1.)
        self.betaI = Default(betaI, .34)
        self.betaP = Default(betaP, .09)
        # photosynthetic thermal dependency
        self.Ea = Default(Ea, 6e4)
        self.R = Default(R, 8.31446261815324)
        self.Kvar = Default(Kvar, 2.45)
        self.nn = Default(nn, 60)
        # photosynthetic flow dependency
        self.Pumin = Default(Pumin, .68886964)
        if processes.fme:
            self.ucr = Default(ucr, .17162374)
        else:
            self.ucr = Default(ucr, .5173)
        # population dynamics
        self.rG = Default(rG, .002)
        self.rR = Default(rR, .2)
        self.rM = Default(rM, .04)
        self.rB = Default(rB, 8.)
        # calcification
        self.gC = Default(gC, .5)
        self.omegaA0 = Default(omegaA0, 5.)
        self.omega0 = Default(omega0, .14587415)
        self.kappaA = Default(kappaA, .66236107)
        # morphological development
        self.Xf = Default(Xf, .1)
        self.Xp = Default(Xp, .5)
        self.Xpu = Default(Xpu, .1)
        self.Xs = Default(Xs, .5/np.sqrt(2.))
        self.XsI = Default(XsI, .1)
        self.Xsu = Default(Xsu, .1)
        self.u0 = Default(u0, .2)
        self.rhoc = Default(rhoc, 1600.)
        # dislodgement criterion
        self.sigmat = Default(sigmat, 2e5)
        self.Cd = Default(Cd, 1.)
        self.rhow = Default(rhow, 1025.)
        # coral recruitment
        self.NoLarvae = Default(NoLarvae, 1e6)
        self.probSett = Default(probSett, 1e-4)
        self.dLarv = Default(dLarv, 1e-3)

constants = Constants()

# =============================================================================
# # # # environmental conditions
# =============================================================================

class Environment():
    
    def __init__(self, light=None, LAC=None, temperature=None, acidity=None,
                 stormcat=None):
        self.light = light
        self.Kd = LAC
        self.temp = temperature
        self.acid = acidity
        self.stormcat = stormcat
        
    @property
    def tempK(self):
        """Temperature in Kelvin."""
        if all(self.temp) < 100.:
            return self.temp + 273.15
        else:
            return self.temp
    
    @property
    def tempC(self):
        """Temperature in Celsius."""
        if all(self.temp) > 100.:
            return self.temp - 273.15
        else:
            return self.temp
    
    @property
    def tempMMM(self):
        MM = self.tempK.groupby([self.tempK.index.year,
                                 self.tempK.index.month]).agg(['mean'])
        MMM = MM.groupby(level=0).agg(['min', 'max'])
        MMM.columns = MMM.columns.droplevel([0, 1])
        return MMM
    
    @property
    def dates(self):
        d = self.temp.reset_index().drop('sst', axis=1)
        return pd.to_datetime(d['date'])
    
    def fromFile(self, param, file, fdir=None):
        def date2index(self):
            """Function applicable to time-series in Pandas."""
            self['date'] = pd.to_datetime(self['date'])
            self.set_index('date', inplace=True)
        if fdir is None:
            f = file
        else:
            f = os.path.join(fdir, file)
        
        if param == 'light':
            self.light = pd.read_csv(f, sep='\t')
            date2index(self.light)
        elif param == 'LAC':
            self.Kd = pd.read_csv(f, sep='\t')
            date2index(self.Kd)
        elif param == 'temperature':
            self.temp = pd.read_csv(f, sep='\t')
            date2index(self.temp)
        elif param == 'acidity':
            self.acid = pd.read_csv(f, sep='\t')
            date2index(self.acid)
        elif param == 'storm':
            self.stormcat = pd.read_csv(f, sep='\t')
            self.stormcat.set_index('year', inplace=True)

# =============================================================================
# # # # coral definition
# =============================================================================
        
class Morphology():

    def __init__(self, dc, hc, bc, tc, ac):
        """
        Definition of morphology.

        Parameters
        ----------
        dc : numeric
            Diameter of the plate of the coral [m].
        hc : numeric
            Coral height [m].
        bc : numeric
            Diameter of the base of the coral [m].
        tc : numeric
            Thickness of the plate of the coral [m].
        ac : numeric
            Axial distance between corals [m].

        Returns
        -------
        Coral morphology.

        """
        self.dc = dc
        self.hc = hc
        self.bc = bc
        self.tc = tc
        self.ac = ac
    
    def __repr__(self):
        return "Morphology({}, {}, {}, {}, {})".format(
            self.dc, self.hc, self.bc, self.tc, self.ac)
    
    def __str__(self):
        return ('Coral morphology with: '
                'dc = {}; hc = {}, bc = {}, tc = {}, ac = {}'
                .format(self.dc, self.hc, self.bc, self.tc, self.ac))
    
    @property
    def rf(self):
        return self.hc / self.dc
    
    @property
    def rp(self):
        return self.bc / self.dc
    
    @property
    def rs(self):
        return self.dc / self.ac
    
    @property
    def dcRep(self):
        return (self.bc * (self.hc - self.tc) + self.dc * self.tc) / self.hc
    
    @property
    def volume(self):
        return .25 * np.pi * (
            (self.hc - self.tc) * self.bc ** 2 + self.tc * self.dc ** 2)
    
    @property
    def dcMatrix(self):
        M = DataReshape(spacetime)
        return M.param2matrix(self.dc, 'space')
    
    @property
    def hcMatrix(self):
        M = DataReshape(spacetime)
        return M.param2matrix(self.hc, 'space')
    
    @property
    def bcMatrix(self):
        M = DataReshape(spacetime)
        return M.param2matrix(self.bc, 'space')
    
    @property
    def tcMatrix(self):
        M = DataReshape(spacetime)
        return M.param2matrix(self.tc, 'space')
    
    @property
    def acMatrix(self):
        M = DataReshape(spacetime)
        return M.param2matrix(self.ac, 'space')
    
    @property
    def dcRepMatrix(self):
        M = DataReshape(spacetime)
        return M.param2matrix(self.dcRep, 'space')
    
    @volume.setter
    def volume(self, volume):
        def Vc2dc(self):
            """Coral volume > diameter of the plate."""
            self.dc = ((4. * volume) / (np.pi * rf * rp * (
                1. + rp - rp ** 2))) ** (1. / 3.)
        def Vc2hc(self):
            """Coral volume > coral height."""
            self.hc = ((4. * volume * rf ** 2) / (np.pi * rp * (
                1. + rp - rp ** 2))) ** (1. / 3.)
        def Vc2bc(self):
            """Coral volume > diameter of the base."""
            self.bc = ((4. * volume * rp ** 2) / (np.pi * rf * (
                1. + rp - rp ** 2))) ** (1. / 3.)
        def Vc2tc(self):
            """Coral volume > thickness of the plate."""
            self.tc = ((4. * volume * rf ** 2 * rp ** 2) / (np.pi * (
                1. + rp - rp ** 2))) ** (1. / 3.)
        def Vc2ac(self):
            """Coral volume > axial distance."""
            self.ac = (1. / rs) * ((4. * volume) / (np.pi * rf * rp * (
                1. + rp - rp ** 2))) ** (1. / 3.)
        # # obtain previous morphological ratios
        rf = self.rf
        rp = self.rp
        rs = self.rs
        # # update morphology
        Vc2dc(self)
        Vc2hc(self)
        Vc2bc(self)
        Vc2tc(self)
        Vc2ac(self)
    
    
    def morph2vegden(self):
        """
        Translation from morphological dimensions to vegetation density.
        """
        try:
            rnveg = np.zeros(self.ac.shape)
            rnveg[self.ac > 0.] = (
                2. * self.dcRep[self.ac > 0.]) / (self.ac[self.ac > 0.] ** 2)
        except TypeError:
            if self.ac > 0.:
                rnveg = (2. * self.dcRep) / (self.ac ** 2)
            else:
                rnveg = 0.
        return rnveg
            
    
    def plot(self, axLabels=False, axTicks=False, explanation=False,
             save=False, figname=None, figdir=None):

        def outerLines(self, x0):
            x = [x0 - .5 * self.bc,
                 x0 - .5 * self.bc,
                 x0 - .5 * self.dc,
                 x0 - .5 * self.dc,
                 x0 + .5 * self.dc,
                 x0 + .5 * self.dc,
                 x0 + .5 * self.bc,
                 x0 + .5 * self.bc]
            y = [0.,
                 self.hc - self.tc,
                 self.hc - self.tc,
                 self.hc,
                 self.hc,
                 self.hc - self.tc,
                 self.hc - self.tc,
                 0.]
            return x, y

        def annotateText(ax, xyfrom, xyto, text=None):
            if text is None:
                text = str(np.sqrt((xyfrom[0] - xyto[0]) ** 2 +
                                   (xyfrom[1] - xyto[1]) ** 2))
            ax.annotate(
                '', xyfrom, xyto,
                arrowprops=dict(arrowstyle='<->'))
            ax.text(
                (xyto[0] + xyfrom[0]) / 2., .01 + (xyto[1] + xyfrom[1]) / 2.,
                text, fontsize=12)

        left = outerLines(self, 0.)
        right = outerLines(self, self.ac)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        # zero-line
        ax.plot([-.5 * self.ac, 1.5 * self.ac], [0., 0.],
                color='gray', alpha=1.,
                linewidth=2., linestyle='solid',
                label='_nolegend_')
        # plot data
        ax.plot(left[0], left[1],
                color='black', alpha=1.,
                linewidth=1., linestyle='solid',
                label='_nolegend_')
        ax.plot(right[0], right[1],
                color='gray', alpha=.5,
                linewidth=1., linestyle='dashed',
                label='_nolegend_')
        # axes labels
        if axLabels:
            ax.set_xlabel('horizontal distance [m]')
            ax.set_ylabel('vertical distance [m]')
        # axes ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if not axTicks:
            plt.tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)
            plt.tick_params(
                axis='y',
                which='both',
                left=False,
                right=False,
                labelleft=False)
        # plot limits
        ax.set_xlim([-.5 * self.ac, 1.5 * self.ac])
        ax.set_ylim([0., 1.2 * self.hc])
        # explaining lines / texts / etc.
        if explanation:
            # dc
            annotateText(
                ax, (-.5 * self.dc, 1.15 * self.hc),
                (.5 * self.dc, 1.15 * self.hc),
                text=r'$d_{{c}}$')
            # hc
            annotateText(
                ax, (-.25 * self.ac, 0.), (-.25 * self.ac, self.hc),
                text=r'$h_{{c}}$')
            # bc
            annotateText(
                ax, (-.5 * self.bc, .5 * (self.hc - self.tc)),
                (.5 * self.bc, .5 * (self.hc - self.tc)),
                text=r'$b_{{c}}$')
            # tc
            annotateText(
                ax, (.25 * self.ac, self.hc - self.tc),
                (.25 * self.ac, self.hc),
                text=r'$t_{{c}}$')
            # ac
            annotateText(
                ax, (0., 1.05 * self.hc), (self.ac, 1.05 * self.hc),
                text=r'$a_{{c}}$')
        # legend / title
        
        # # save figure
        if save:
            if figname is None:
                print('WARNING: No figure name specified.')
                figname = 'Morphology'
            figfile = figname + '.png'
            if figdir is None:
                figfull = figfile
            else:
                figfull = os.path.join(figdir, figfile)
            fig.savefig(figfull, dpi=300, bbox_inches='tight')
            print('Figure saved as: "{}"'.format(figfull))

# =============================================================================
# # # # biophysical relations
# =============================================================================

class Light():
    
    def __init__(self, I0, Kd, h):
        """
        Light micro-environment.

        Parameters
        ----------
        I0 : numeric
            Incoming light-intensity as measured at water-air interface
        [umol photons m-2 s-1].
        Kd : numeric
            Light-attenuation coefficient [m-1].
        h : numeric
            Water depth (excl. coral canopy) [m].

        """
        M = DataReshape(spacetime)
        self.I0 = M.param2matrix(I0, 'time')
        self.Kd = M.param2matrix(Kd, 'time')
        self.h = M.param2matrix(h, 'space')
    
    def repLight(self, coral):
        """Representative light-intensity."""
        Light.biomass(self, coral)
        # light catchment top of plate
        top = .25 * np.pi * coral.dcMatrix ** 2 * self.I0 * np.exp(
            - self.Kd * (self.h - coral.hcMatrix))
        # light catchment side of plate
        side = (np.pi * coral.dcMatrix * self.I0) / self.Kd * (
            np.exp(- self.Kd * (self.h - coral.hcMatrix)) -
            np.exp(- self.Kd * (self.h - coral.hcMatrix + coral.tcMatrix)))
        # light catchment side of base
        base = (np.pi * coral.bcMatrix * self.I0) / self.Kd * (
            np.exp(- self.Kd * (self.h - self.L)) -
            np.exp(- self.Kd * self.h))
        # total light catchment
        total = top + side + base
        # biomass-averaged
        try:
            coral.light = self.I0 * np.exp(- self.Kd * self.h)
            coral.light[coral.Bc > 0.] = (
                total[coral.Bc > 0.] / coral.Bc[coral.Bc > 0.])
        except TypeError:
            if coral.Bc > 0.:
                coral.light = total / coral.Bc
            else:
                coral.light = self.I0 * np.exp(- self.Kd * self.h)
    
    def biomass(self, coral):
        """Coral biomass."""
        Light.baseLight(self, coral)
        coral.Bc = np.pi * (.25 * coral.dcMatrix ** 2 +
                            coral.dcMatrix * coral.tcMatrix +
                            coral.bcMatrix * self.L)
    
    def baseLight(self, coral):
        def lightSpreading(self):
            theta = constants.thetamax * np.exp(
                - self.light_attenuation * (self.h - coral.hcMatrix + coral.tcMatrix))
            return theta
        theta = lightSpreading(self)
        self.L = coral.hcMatrix - coral.tcMatrix - (
            coral.dcMatrix - coral.bcMatrix) / (
            2. * np.tan(.5 * theta))
        try:
            self.L[self.L < 0.] = 0.
        except TypeError:
            if self.L < 0.:
                self.L = 0.
    
class Flow():
    
    def __init__(self, Ucurr, Uwave, h, Tp):
        """
        Flow micro-environment.

        Parameters
        ----------
        Ucurr : numeric
            Current-induced flow velocity [m s-1].
        Uwave : numeric
            Wave-induced flow velocity [m s-1].
        h : numeric
            Water depth (excl. coral canopy) [m].
        Tp : numeric
            Peak wave period [s].

        """
        self.uc = DataReshape.param2array(Ucurr)
        self.uw = DataReshape.param2array(Uwave)
        self.h = DataReshape.param2array(h)
        self.Tp = DataReshape.param2array(Tp)
    
    def waveCurrent(self, coral, incanopy=True):
        """Wave-current interaction."""
        if incanopy:
            alphaw = self.waveAttenuation(coral, wac='wave')
            alphac = self.waveAttenuation(coral, wac='current')
            coral.ucm = np.sqrt((alphaw * self.uw) ** 2 +
                               (alphac * self.uc) ** 2 +
                               (2. * alphaw * self.uw * alphac * self.uc *
                                np.cos(constants.wcAngle)))
        else:
            coral.ucm = np.sqrt(self.uw ** 2 + self.uc ** 2 +
                                2. * self.uw * self.uc *
                                np.cos(constants.wcAngle))
        coral.um = np.sqrt(self.uw ** 2 + self.uc ** 2 +
                           2. * self.uw * self.uc *
                           np.cos(constants.wcAngle))
        
    
    def waveAttenuation(self, coral, wac):
        """Wave-attenuation coefficient."""
        # # function and derivative defintions
        def func(beta):
            # plt.scatter(beta.real, beta.imag, alpha=.5)
            # components
            shear = (8. * af) / (3. * np.pi * Lshear[i]) * (
                abs(1. - beta) * (1. - beta))
            drag = (8. * af) / (3. * np.pi * Ldrag) * (abs(beta) * beta)
            inertia = 1j * beta * (
                (constants.Cm * lambdap[i]) / (1. - lambdap[i]))
            # combined
            f = 1j * (beta - 1.) - shear + drag + inertia
            # output
            return f
        
        def deriv(beta):
            # components
            shear = (((1. - beta) ** 2 / abs(1. - beta) - abs(1. - beta)) /
                     Lshear[i])
            drag = (beta ** 2 / abs(beta) + beta) / Ldrag
            inertia = 1j * (constants.Cm * lambdap[i]) / (1. - lambdap[i])
            # combined
            df = 1j + (8. * af) / (3. * np.pi) * (- shear + drag) + inertia
            # output
            return df
        
        # # parameter definitions
        # wave- vs. current-induced
        if wac == 'wave':
            T = self.Tp
            U = self.uw
        elif wac == 'current':
            T = 1e3 * np.ones(spacetime[0])
            U = self.uc
        # geometric parameters
        Ap = .25 * np.pi * coral.dcRep ** 2
        Af = coral.dcRep * coral.hc
        AT = .5 * coral.ac ** 2
        lambdap = Ap / AT
        lambdaf = Af / AT
        Lshear = coral.hc / (constants.Cs ** 2)
        
        # # calculations - single layer canopy
        if spacetime[0] == 1:
            lambdap = [lambdap]
            lambdaf = [lambdaf]
            Lshear = [Lshear]
        aw = np.ones(spacetime[0])
        for i in range(spacetime[0]):
            # no corals
            if coral.K[i] == 0.:
                aw[i] = 1.
            # emergent
            elif self.h[i] <= coral.hc[i]:
                up = U[i] / (1. - lambdap[i])
                uc = ((1. - lambdap[i]) / (1. - np.sqrt(4. * lambdap[i]) /
                       (constants.psi * np.pi))) * up
                Re = (uc * coral.dcRep[i]) / constants.nu
                Cd = 1. + 10. * Re ** (- 2. / 3.)
                aw[i] = 1.
            # submerged
            else:
                # initial values for iteration
                uf = U[i]
                Cd = 1.
                # iteration
                for k in range(int(constants.maxiter_k)):
                    Ldrag = (2. * coral.hc[i] * (1. - lambdap[i])) / (
                        Cd * lambdaf[i])
                    af = (uf * T[i]) / (2. * np.pi)
                    if wac == 'wave':
                        aw[i] = abs(newton(
                            func, x0=complex(.5, .5), fprime=deriv,
                            maxiter=constants.maxiter_aw))
                    elif wac == 'current':
                        X = Ldrag / Lshear[i] * (
                            coral.hc[i] / (self.h[i] - coral.hc[i]) + 1.)
                        aw[i] = (X - np.sqrt(X)) / (X - 1.)
                    up = aw[i] * uf
                    uc = (1. - lambdap[i]) / (
                        1. - np.sqrt((4. - lambdap[i]) / (
                            constants.psi * np.pi))) * up
                    Re = (uc * coral.dcRep[i]) / constants.nu
                    Cdk = 1. + 10. * Re ** (-2. / 3.)
                    if abs((Cdk - Cd) / Cdk) <= constants.err:
                        break
                    else:
                        Cd = float(Cdk)
                        uf = abs((1. - constants.numericTheta) * uf +
                                 constants.numericTheta * (
                                     self.h[i] * U[i] - coral.hc[i] * up) /
                                 (self.h[i] - coral.hc[i]))
                    if k == constants.maxiter_k:
                        print(
                            'WARNING: maximum number of iterations reached: {}'
                            .format(constants.maxiter_k))
        return aw
                    
    def TBL(self, coral):
        """Thermal boundary layer."""
        Flow.VBL(self, coral)
        coral.deltat = self.delta * (
            (constants.alpha / constants.nu) ** (1. / 3.))
    
    def VBL(self, coral):
        """Velocity boundary layer."""
        try:
            self.delta = np.zeros(coral.ucm.shape)
            self.delta[coral.ucm > 0.] = (
                (constants.rd * constants.nu) / (
                    np.sqrt(constants.Cf) * coral.ucm[coral.ucm > 0.]))
        except TypeError:
            if coral.ucm > 0.:
                self.delta = (constants.rd * constants.nu) / (
                    np.sqrt(constants.Cf) * coral.ucm)
            else:
                self.delta = 0.

class Temperature():
    
    def __init__(self, T, MMM=None):
        """
        Thermal micro-environment.

        Parameters
        ----------
        h : numeric
            Water depth (excl. coral canopy) [m].
        T : numeric
            Temperature of water [K].
        MMM : numeric
            Thermal maximum monthly mean [K].

        """
        M = DataReshape(spacetime)
        self.T = M.param2matrix(T, 'time')
        self.MMM = MMM
    
    def coralTemperature(self, coral):
        """Coral temperature."""
        M = DataReshape(spacetime)
        deltat = M.param2matrix(coral.deltat, 'space')
        if processes.tme:
            coral.dTc = ((deltat * constants.ap) /
                         (constants.k * constants.K0)) * coral.light
            coral.temp = self.T + coral.dTc
        else:
            coral.temp = self.T

class Photosynthesis():
    
    def __init__(self, I0, firstYear=True):
        """
        Photosynthetic efficiency based on photosynthetic dependencies.

        Parameters
        ----------
        I0 : numeric
            Incoming light-intensity as measured at water-air interface
            [umol photons m-2 s-1].
        firstYear : bool, optional
            First year of the model simulation. The default is True.

        """
        M = DataReshape(spacetime)
        self.I0 = M.param2matrix(I0, 'time')
        self.firstYear = firstYear
    
    def photosyn(self, coral, environment, year):
        """Photosynthetic efficiency."""
        M = DataReshape(spacetime)
        # components
        Photosynthesis.PLD(self, coral, 'qss')
        Photosynthesis.PTD(self, coral, environment, year)
        Photosynthesis.PFD(self, coral)
        # combined
        if processes.pfd:
            coral.photo_rate = self.pld * self.ptd * M.param2matrix(
                self.pfd, 'space')
        else:
            coral.photo_rate = self.pld * self.ptd
    
    def PLD(self, coral, output):
        """Photosynthetic light dependency."""
        def photoacc(self, Xold, param):
            """Photo-acclimation."""
            # # parameter definition
            if param == 'Ik':
                Xmax = constants.Ikmax
                betaX = constants.betaI
            elif param == 'Pmax':
                Xmax = constants.Pmmax
                betaX = constants.betaP
            # # calculations
            XS = Xmax * (coral.light / self.I0) ** betaX
            if output =='qss':
                return XS
            elif output == 'new':
                Xnew = XS + (Xold - XS) * np.exp(- constants.iota)
                return Xnew
        # # parameter definition
        if output == 'qss':
            Ik = photoacc(self, None, 'Ik')
            Pmax = photoacc(self, None, 'Pmax')
        elif output == 'new':
            NotImplemented
        # # calculations
        self.pld = Pmax * (np.tanh(coral.light / Ik) -
                           np.tanh(.01 * self.I0 / Ik))
        
    
    def PTD(self, coral, environment, year):
        """Photosynthetic thermal dependency."""
        def thermacc(self):
            """Thermal-acclimation."""
            if processes.tme:
                if self.firstYear:
                    environment.tmeMMMmin = pd.DataFrame(
                        data=pd.concat(
                            [environment.tempMMM['min']]*spacetime[0],
                            axis=1).values,
                        columns=np.arange(spacetime[0])) + coral.dTc
                    environment.tmeMMMmax = pd.DataFrame(
                        data=pd.concat(
                            [environment.tempMMM['max']]*spacetime[0],
                            axis=1).values,
                        columns=np.arange(spacetime[0])) + coral.dTc
                else:
                    environment.tmeMMMmin[
                        environment.tmeMMM.index == year] += coral.dTc
                    environment.tmeMMMmax[
                        environment.tmeMMM.index == year] += coral.dTc

                MMMmin = environment.tmeMMMmin[np.logical_and(
                    environment.tmeMMM.index < year,
                    environment.tmeMMM.index >= year - int(
                        constants.nn / coral.Csp))]
                mmin = MMMmin.mean(axis=0)
                smin = MMMmin.std(axis=0)

                MMMmax = environment.tmeMMMmax[np.logical_and(
                    environment.tmeMMM.index < year,
                    environment.tmeMMM.index >= year - int(
                        constants.nn / coral.Csp))]
                mmax = MMMmax.mean(axis=0)
                smax = MMMmax.std(axis=0)
            else:
                MMM = environment.tempMMM[np.logical_and(
                    environment.tempMMM.index < year,
                    environment.tempMMM.index >= year - int(
                        constants.nn / coral.Csp))]
                mmin, mmax = MMM.mean(axis=0)
                smin, smax = MMM.std(axis=0)
            coral.Tlo = mmin - constants.Kvar * smin
            coral.Thi = mmax + constants.Kvar * smax
        
        def adaptTemp(self, coral):
            """Adapted temperature response."""

            def spec(self):
                """Specialisation term."""
                self.spec = 4e-4 * np.exp(-.33 * (self.DT - 10.))

            spec(self)
            self.f1 = - ((coral.temp - coral.Tlo) *
                         ((coral.temp - coral.Tlo) ** 2 - self.DT ** 2))
            Tcr = coral.Tlo - (1. / np.sqrt(3.)) * self.DT
            try:
                if processes.tme:
                    self.f1[coral.temp <= Tcr] = - (
                        (2. / (3. * np.sqrt(3.))) *
                        self.DT[coral.temp <= Tcr] ** 3)
                else:
                    self.f1[coral.temp <= Tcr] = -(
                        (2. / (3. * np.sqrt(3.))) * self.DT ** 3)
            except TypeError:
                if coral.temp <= Tcr:
                    self.f1 = (2. / (3. * np.sqrt(3.))) * self.DT ** 3
            self.f1 *= self.spec
        
        def thermEnv(self):
            """Thermal envelope."""
            self.f2 = np.exp(
                (constants.Ea / constants.R) * (1. / 300. - 1. / self.Topt))
        
        # # parameter definition
        thermacc(self)
        self.DT = coral.Thi - coral.Tlo
        self.Topt = coral.Tlo + (1. / np.sqrt(3.)) * self.DT
        # # calculations
        # components
        adaptTemp(self, coral)
        thermEnv(self)
        # combined
        self.ptd = self.f1 * self.f2
        
    
    def PFD(self, coral):
        """Photosynthetic flow dependency."""
        self.pfd = constants.Pumin + (1. - constants.Pumin) * np.tanh(
            2. * coral.ucm / constants.ucr)


class PopStates():
    
    def __init__(self):
        """
        Population dynamics.
        """
        
    def popstates_t(self, coral, dt=1.):
        """Population dynamics over time."""
        coral.pop_states = np.zeros((spacetime[0], spacetime[1], 4))
        photosynthesis = np.zeros(spacetime[0])
        for n in range(spacetime[1]):
            photosynthesis[coral.K > 0] = coral.photo_rate[coral.K > 0, n]
            coral.pop_states[:, n, :] = self.popstates_xy(
                    coral, photosynthesis, dt=dt)
            coral.p0[coral.K > 0, :] = coral.pop_states[coral.K > 0, n, :]
    
    def popstates_xy(self, coral, PS, dt=1.):
        """Population dynamics over space."""
        P = np.zeros((spacetime[0], 4))
        # # calculations
        # growing conditions
        # > bleached pop.
        P[PS > 0., 3] = (
                coral.p0[PS > 0., 3] / (1. + dt * (
                8. * constants.rR * PS[PS > 0.] /
                coral.Csp + constants.rM * coral.Csp)))
        # > pale pop.
        P[PS > 0., 2] = (
            (coral.p0[PS > 0., 2] + (
                8. * dt * constants.rR * PS[PS > 0.] / coral.Csp) *
             P[PS > 0., 3]) / (
                 1. + dt * constants.rR * PS[PS > 0.] * coral.Csp))
        # > recovering pop.
        P[PS > 0., 1] = (
            (coral.p0[PS > 0., 1] + dt * constants.rR * PS[PS > 0.] *
             coral.Csp * P[PS > 0., 2]) /
            (1. + .5 * dt * constants.rR * PS[PS > 0.] * coral.Csp))
        # > healthy pop.
        a = (dt * constants.rG * PS[PS > 0.] * coral.Csp /
             coral.K[PS > 0.])
        b = (1. - dt * constants.rG * PS[PS > 0.] * coral.Csp * (
                1. - P[PS > 0., 1:].sum(axis=1) /
                coral.K[PS > 0.]))
        c = - (coral.p0[PS > 0., 0] +
               .5 * dt * constants.rR * PS[PS > 0.] *
               coral.Csp * P[PS > 0., 1])
        P[PS > 0., 0] = (
            -b + np.sqrt(b ** 2 - 4 * a * c)) / (2. * a)
        
        # bleaching conditions
        # > healthy pop.
        P[PS <= 0., 0] = (
                coral.p0[PS <= 0., 0] / (
                1. - dt * constants.rB * PS[PS <= 0.] * coral.Csp))
        # > recovering pop.
        P[PS <= 0., 1] = (
                coral.p0[PS <= 0., 1] / (
                1. - dt * constants.rB * PS[PS <= 0.] * coral.Csp))
        # > pale pop.
        P[PS <= 0., 2] = (
            (coral.p0[PS <= 0., 2] - dt * constants.rB *
             PS[PS <= 0.] * coral.Csp * (
                 P[PS <= 0., 0] + P[PS <= 0., 1])) /
            (1. - .5 * dt * constants.rB * PS[PS <= 0.] * coral.Csp))
        # > bleached pop.
        P[PS <= 0., 3] = (
            (coral.p0[PS <= 0., 3] -
             .5 * dt * constants.rB * PS[PS <= 0.] * coral.Csp *
             P[PS <= 0., 2]) /
            (1. - .25 * dt * constants.rB * PS[PS <= 0.] * coral.Csp))
        
        # # check on carrying capacity
        if any(P.sum(axis=1) > 1.0001 * coral.K):
            print('WARNING: total population than carrying capacity at {}. '
                  '(PT = {}, K = {})'
                  .format(
                      np.arange(
                          len(coral.K))[P.sum(axis=1) > 1.0001 * coral.K],
                      P[P.sum(axis=1) > 1.0001 * coral.K],
                      coral.K[P.sum(axis=1) > 1.0001 * coral.K]))
        
        # # output
        return P

class Calcification():
    
    def __init__(self):
        """
        Calcification rate.
        """
    
    def calRate(self, coral, omega):
        """Calcification rate."""

        def aragoniteDependency(self):
            """Aragonite dependency."""
            self.ad = (omega - constants.omega0) / (
                constants.kappaA + omega - constants.omega0)
            M = DataReshape(spacetime)
            self.ad = M.param2matrix(self.ad, 'time')
        
        aragoniteDependency(self)
        coral.calc = (
                constants.gC * coral.Csp * coral.pop_states[:, :, 0] *
                self.ad * coral.photo_rate)

class MorDevelopment():
    
    def __init__(self, Gsum, h, Kd, dtyear=1.):
        """
        Morphological development.

        Parameters
        ----------
        Gsum : numeric
            Accumulation of calcification of [dtyear] years [kg m-2 yr-1].
        h : numeric
            Water depth (excl. coral canopy) [m].
        Kd : numeric
            Light-attenuation coefficient [m-1].
        dtyear : float, optional
            Update interval [yr]. The default is 1.

        """
        M = DataReshape(spacetime)
        self.Gsum = Gsum
        self.h = M.param2matrix(h, 'space')
        self.Kd = M.param2matrix(Kd, 'time')
        self.dtyear = dtyear
    
    def update(self, coral, I0):
        """Update morphology."""
        M = DataReshape(spacetime)
        self.I0 = M.param2matrix(I0, 'time')
        # # optimal morphological ratios
        def rfOptimal(self):
            """Optimal form ratio."""
            self.rfOpt = constants.Xf * (
                    coral.light.mean(axis=1) / self.I0.mean(axis=1)) / (
                            constants.u0 / 1e-6)
            self.rfOpt[coral.ucm > 0.] = constants.Xf * (
                coral.light.mean(axis=1)[coral.ucm > 0.] / 
                self.I0.mean(axis=1)[coral.ucm > 0.]) / (
                constants.u0 / coral.ucm[coral.ucm > 0.])
        def rpOptimal(self):
            """Optimal plate ratio."""
            self.rpOpt = constants.Xp * (1. + np.tanh(
                constants.Xpu * (coral.ucm - constants.u0) / constants.u0))
        def rsOptimal(self):
            """Optimal spacing ratio."""
            self.rsOpt = constants.Xs * (
                1. - np.tanh(
                    constants.XsI * coral.light.mean(axis=1) /
                    self.I0.mean(axis=1))) * (
                    1. + np.tanh(constants.Xsu * (coral.ucm - constants.u0) /
                                 constants.u0))
        
        # # increased coral volume
        def deltaVolume(self):
            """Volumetric change."""
            self.dVc = (
                               .5 * coral.ac ** 2 * self.calc_sum * self.dt_year /
                               constants.rhoc) * coral.Bc.mean(axis=1)
        
        # # update morphological ratio
        def ratioUpdate(self, ratio):
            """Update morphological ratios."""
            # input check
            ratios = ['rf', 'rp', 'rs']
            if ratio not in ratios:
                raise ValueError(
                    'Morphological ratio not in {}'
                    .format(ratios))
            # calculations
            deltaVolume(self)
            def PDE(self, rold, ropt):
                """Mass balance."""
                r = (coral.volume * rold + self.dVc * ropt) / (
                    coral.volume + self.dVc)
                return r
            if ratio == 'rf':
                rfOptimal(self)
                self.rf = PDE(self, coral.rf, self.rfOpt)
            elif ratio == 'rp':
                rpOptimal(self)
                self.rp = PDE(self, coral.rp, self.rpOpt)
            elif ratio == 'rs':
                rsOptimal(self)
                self.rs = PDE(self, coral.rs, self.rsOpt)
        
        # # calculations
        # update ratios
        ratios = ['rf', 'rp', 'rs']
        for r in ratios:
            ratioUpdate(self, r)
        # update morphology
        coral.volume += self.dVc


class Dislodgement():
    
    def __init__(self):
        """
        Dislodgement check.
        """
    
    def update(self, coral, survivalCoefficient=1.):
        """Update morphology due to storm damage."""
        # # partial dislodgement
        Dislodgement.partialDislodgement(self, coral, survivalCoefficient)
        # # update
        # population states
        for s in range(4):
            coral.p0[:, s] *= self.pardis
        # morphology
        coral.volume *= self.pardis
    
    def partialDislodgement(self, coral, survivalCoefficient=1.):
        """Percentage surviving storm event."""
        try:
            self.pardis = np.ones(coral.dc.shape)
            dislodged = Dislodgement.dislodgementCriterion(self, coral)
            self.pardis[dislodged] = survivalCoefficient * (
                    self.dmt[dislodged] / self.csf[dislodged])
        except TypeError:
            if Dislodgement.dislodgementCriterion(self, coral):
                self.pardis = survivalCoefficient * self.dmt / self.csf
            else:
                self.pardis = 1.
    
    def dislodgementCriterion(self, coral):
        """Dislodgement criterion. Returns boolean (array)."""
        Dislodgement.DMT(self, coral)
        Dislodgement.CSF(self, coral)
        return self.dmt <= self.csf
    
    def DMT(self, coral):
        """Dislodgement Mechanical Threshold."""
        try:
            self.dmt = 1e20 * np.ones(coral.um.shape)
            self.dmt[coral.um > 0] = constants.sigmat / (
                constants.rhow * constants.Cd * coral.um[coral.um > 0] ** 2)
        except TypeError:
            if coral.um > 0:
                self.dmt = constants.sigmat / (
                    constants.rhow * constants.Cd * coral.um ** 2)
            else:
                self.dmt = 1e20
    
    def CSF(self, coral):
        """Colony Shape Factor."""
        # arms of moment
        at = coral.hc - .5 * coral.tc
        ab = .5 * (coral.hc - coral.tc)
        # area of moment
        At = coral.dc * coral.tc
        Ab = coral.bc * (coral.hc - coral.tc)
        # integral
        S = at * At + ab * Ab
        # colony shape factor
        try:
            self.csf = np.zeros(coral.dc.shape)
            self.csf[coral.bc > 0] = 16. / (np.pi * coral.bc ** 3) * S
        except TypeError:
            if coral.bc > 0:
                self.csf = 16. / (np.pi * coral.bc ** 3) * S
            else:
                self.csf = 0.

class Recruitment():
    
    def __init__(self):
        """
        Recruitment dynamics.
        """
    
    def update(self, coral):
        """Update coral cover / volume after spawning event."""
        coral.p0[:, 0] += Recruitment.spawning(self, coral, 'P')
        coral.volume += Recruitment.spawning(self, coral, 'V')
    
    def spawning(self, coral, param):
        """Contribution due to mass coral spawning."""
        # # input check
        params = ['P', 'V']
        if param not in params:
            raise ValueError(
                'Parameter definition not in {}'
                .format(params))
        
        # # calculations
        # potential
        if param == 'P':
            S = constants.probSett * constants.NoLarvae * constants.dLarv ** 2
        elif param == 'V':
            S = constants.probSett * constants.NoLarvae * constants.dLarv ** 3
        # recruitment
        self.PHav = coral.pop_states[:, -1, 0].mean()
        recr = np.zeros(coral.K.shape)
        recr[coral.K > 0] = S * self.PHav * (
                1. - coral.pop_states[coral.K > 0, -1, :].sum(axis=1) /
                coral.K[coral.K > 0])
        
        # # output
        return recr

print('Biophysical relations defined.\n')

# =============================================================================
# # # # write output-files
# =============================================================================

def outputMap(coral, mapLME, mapFME, mapTME, mapPD, mapPS, mapC, mapMD,
              year, firstYear, mapFile):
    """
    Write data to NetCDF4-file is mapped format.

    Parameters
    ----------
    mapLME : bool
        Write light micro-environment; i.e. representative light-intensity
        (annual mean).
    mapFME : bool
        Write flow micro-environment; i.e. in-canopy flow velocity (annual
        mean).
    mapTME : bool
        Write thermal micro-environment; i.e. coral temperature (annual mean).
    mapPD : bool
        Write photosynthetic dependency; i.e. photosynthetic rate (proxy,
        annual mean).
    mapPS : bool
        Write population states; i.e. total coral cover, healthy coral cover,
        recovering coral cover, pale coral cover, and bleached coral cover (end
        of the year).
    mapC : bool
        Write calcification; i.e. calcification rate (annual sum).
    mapMD : bool
        Write morphological development; i.e. diameter of plate, coral height,
        base width, thickness of plate, axial distance, and coral volume
        (annual mean).
    year : numeric
        Year of simulation.
    firstYear: bool
        First year of simulation.
    mapFile : string
        Filename (incl. directory); ends with '.nc' (netCDF4-file).

    Returns
    -------
    Output file: map.

    """
    files = [mapFME, mapTME, mapPD, mapPS, mapC, mapMD]
    if any(files):
        if firstYear:
            mset = Dataset(mapFile, 'w', format='NETCDF4')
            mset.description = 'Mapped simulation data of the CoralModel'

            # dimensions
            mset.createDimension('time', None)
            mset.createDimension('nmesh2d_face', int(ndxi))

            # variables
            t = mset.createVariable('time', int, ('time',))
            t.long_name = 'year'
            t.units = 'years since 0 B.C.'

            x = mset.createVariable('mesh2d_x', 'f8', ('nmesh2d_face',))
            x.long_name = 'x-coordinate'
            x.units = 'm'

            y = mset.createVariable('mesh2d_y', 'f8', ('nmesh2d_face',))
            y.long_name = 'y-coordinate'
            y.units = 'm'
            
            if mapLME:
                Lset = mset.createVariable('Iz', 'f8',
                                           ('time', 'nmesh2d_face'))
                Lset.long_name = 'representative light-intensity'
                Lset.units = 'umol photons m^-2 s^-1'
            if mapFME:
                Uset = mset.createVariable('ucm', 'f8',
                                            ('time', 'nmesh2d_face'))
                Uset.long_name = 'in-canopy flow'
                Uset.units = 'm s^-1'
            if mapTME:
                Tset = mset.createVariable('Tc', 'f8',
                                            ('time', 'nmesh2d_face'))
                Tset.long_name = 'coral temperature'
                Tset.units = 'K'

                Tloset = mset.createVariable('Tlo', 'f8',
                                              ('time', 'nmesh2d_face'))
                Tloset.long_name = 'lower thermal limit'
                Tloset.units = 'K'

                Thiset = mset.createVariable('Thi', 'f8',
                                              ('time', 'nmesh2d_face'))
                Thiset.long_name = 'upper thermal limit'
                Thiset.units = 'K'
            if mapPD:
                PSset = mset.createVariable('PS', 'f8',
                                            ('time', 'nmesh2d_face'))
                PSset.long_name = 'annual mean photosynthesis'
                PSset.units = '-'
            if mapPS:
                PTset = mset.createVariable('PT', 'f8',
                                            ('time', 'nmesh2d_face'))
                PTset.long_name = 'total population'
                PTset.units = '-'

                PHset = mset.createVariable('PH', 'f8',
                                            ('time', 'nmesh2d_face'))
                PHset.long_name = 'healthy population'
                PHset.units = '-'

                PRset = mset.createVariable('PR', 'f8',
                                            ('time', 'nmesh2d_face'))
                PRset.long_name = 'recoverying population'
                PRset.units = '-'

                PPset = mset.createVariable('PP', 'f8',
                                            ('time', 'nmesh2d_face'))
                PPset.long_name = 'pale population'
                PPset.units = '-'

                PBset = mset.createVariable('PB', 'f8',
                                            ('time', 'nmesh2d_face'))
                PBset.long_name = 'bleached population'
                PBset.units = '-'
            if mapC:
                Gset = mset.createVariable('G', 'f8',
                                            ('time', 'nmesh2d_face'))
                Gset.long_name = 'calcification'
                Gset.units = 'kg m^-2 y^-1'
            if mapMD:
                DCset = mset.createVariable('dc', 'f8',
                                            ('time', 'nmesh2d_face'))
                DCset.long_name = 'plate diameter'
                DCset.units = 'm'

                HCset = mset.createVariable('hc', 'f8',
                                            ('time', 'nmesh2d_face'))
                HCset.long_name = 'coral height'
                HCset.units = 'm'

                BCset = mset.createVariable('bc', 'f8',
                                            ('time', 'nmesh2d_face'))
                BCset.long_name = 'base diameter'
                BCset.units = 'm'

                TCset = mset.createVariable('tc', 'f8',
                                            ('time', 'nmesh2d_face'))
                TCset.long_name = 'plate thickness'
                TCset.units = 'm'

                ACset = mset.createVariable('ac', 'f8',
                                            ('time', 'nmesh2d_face'))
                ACset.long_name = 'axial distance'
                ACset.units = 'm'
                
                VCset = mset.createVariable('Vc', 'f8',
                                            ('time', 'nmesh2d_face'))
                VCset.long_name = 'coral volume'
                VCset.units = 'm^3'

            # data
            t[:] = np.array([year - 1, year])
            x[:] = xzw[range(ndxi)]
            y[:] = yzw[range(ndxi)]
            if mapLME:
                Lset[:, :] = np.array(
                    [np.zeros(ndxi), coral.light.mean(axis=1)])
            if mapFME:
                Uset[:, :] = np.array([np.zeros(ndxi), coral.ucm])
            if mapTME:
                Tset[:, :] = np.array(
                    [np.zeros(ndxi), coral.temp.mean(axis=1)])
                if processes.tme:
                    Tloset[:, :] = np.array([np.zeros(ndxi), coral.Tlo])
                    Thiset[:, :] = np.array([np.zeros(ndxi), coral.Thi])
                else:
                    Tloset[:, :] = np.array(
                        [np.zeros(ndxi), coral.Tlo * np.ones(ndxi)])
                    Thiset[:, :] = np.array(
                        [np.zeros(ndxi), coral.Thi * np.ones(ndxi)])
            if mapPD:
                PSset[:, :] = np.array(
                    [np.zeros(K.shape), coral.photo_rate.mean(axis=1)])
            if mapPS:
                PTset[:, :] = np.array(
                    [K, coral.pop_states[:, -1, :].sum(axis=1)])
                PHset[:, :] = np.array(
                    [K, coral.pop_states[:, -1, 0]])
                PRset[:, :] = np.array(
                    [np.zeros(K.shape), coral.pop_states[:, -1, 1]])
                PPset[:, :] = np.array([
                    np.zeros(K.shape), coral.pop_states[:, -1, 2]])
                PBset[:, :] = np.array([
                    np.zeros(K.shape), coral.pop_states[:, -1, 3]])
            if mapC:
                Gset[:, :] = np.array(
                    [np.zeros(K.shape), coral.calc.sum(axis=1)])
            if mapMD:
                DCset[:, :] = np.array([dc0 * K, coral.dc])
                HCset[:, :] = np.array([hc0 * K, coral.hc])
                BCset[:, :] = np.array([bc0 * K, coral.bc])
                TCset[:, :] = np.array([tc0 * K, coral.tc])
                ACset[:, :] = np.array([ac0 * K, coral.ac])
                Vc0 = Morphology(dc0, hc0, bc0, tc0, ac0).volume
                VCset[:, :] = np.array([Vc0 * K, coral.volume])
        else:
            mset = Dataset(mapFile, mode='a')
            # append data
            mset['time'][:] = np.append(mset['time'][:], year)
            if mapFME:
                mset['ucm'][-1, :] = coral.ucm
            if mapTME:
                mset['Tc'][-1, :] = coral.temp[:, -1]
                if processes.tme:
                    mset['Tlo'][-1, :] = coral.Tlo
                    mset['Thi'][-1, :] = coral.Thi
                else:
                    mset['Tlo'][-1, :] = coral.Tlo * np.ones(ndxi)
                    mset['Thi'][-1, :] = coral.Thi * np.ones(ndxi)
            if mapPD:
                mset['PS'][-1, :] = coral.photo_rate.mean(axis=1)
            if mapPS:
                mset['PT'][-1, :] = coral.pop_states[:, -1, :].sum(axis=1)
                mset['PH'][-1, :] = coral.pop_states[:, -1, 0]
                mset['PR'][-1, :] = coral.pop_states[:, -1, 1]
                mset['PP'][-1, :] = coral.pop_states[:, -1, 2]
                mset['PB'][-1, :] = coral.pop_states[:, -1, 3]
            if mapC:
                mset['G'][-1, :] = coral.calc.sum(axis=1)
            if mapMD:
                mset['dc'][-1, :] = coral.dc
                mset['hc'][-1, :] = coral.hc
                mset['bc'][-1, :] = coral.bc
                mset['tc'][-1, :] = coral.tc
                mset['ac'][-1, :] = coral.ac
        mset.close()

def outputHis(coral, hisLME, hisFME, hisTME, hisPD, hisPS, hisC, hisMD,
              dates, firstYear, hisFile, date0=None):
    """
    Write data to NetCDF4-file is mapped format.

    Parameters
    ----------
    hisLME : bool
        Write light micro-environment; i.e. representative light-intensity
        (annual mean).
    hisFME : bool
        Write flow micro-environment; i.e. in-canopy flow velocity (annual
        mean).
    hisTME : bool
        Write thermal micro-environment; i.e. coral temperature (annual mean).
    hisPD : bool
        Write photosynthetic dependency; i.e. photosynthetic rate (proxy,
        annual mean).
    hisPS : bool
        Write population states; i.e. total coral cover, healthy coral cover,
        recovering coral cover, pale coral cover, and bleached coral cover (end
        of the year).
    hisC : bool
        Write calcification; i.e. calcification rate (annual sum).
    hisMD : bool
        Write morphological development; i.e. diameter of plate, coral height,
        base width, thickness of plate, axial distance, and coral volume
        (annual mean).
    dates : datetime.datetime
        Dates within year of simulation.
    firstYear: bool
        First year of simulation.
    hisFile : string
        Filename (incl. directory); ends with '.nc' (netCDF4-file).
    date0 : datetime.datetime
        Reference date, similar to start of simulation time used in the
        environmental time-series.

    Returns
    -------
    Output file: his.

    """
    files = [hisLME, hisFME, hisTME, hisPD, hisPS, hisC, hisMD]
    if any(files):
        if date0 is None:
            date0 = datetime.datetime(2000, 1, 1)
        if firstYear:
            hset = Dataset(hisfilef, 'w', format='NETCDF4')
            hset.description = 'Historic simulation data of the CoralModel'

            # dimensions
            hset.createDimension('time', None)
            hset.createDimension('stations', len(xyn))

            # variables
            t = hset.createVariable('time', 'f8', ('time',))
            t.long_name = 'days since January 1, 2000'
            t.units = 'days'

            x = hset.createVariable('station_x_coordinate', 'f8',
                                    ('stations',))
            y = hset.createVariable('station_y_coordinate', 'f8',
                                    ('stations',))
            if hisLME:
                Lset = hset.createVariable('Iz', 'f8',
                                           ('time', 'stations'))
                Lset.long_name = 'representative light-intensity'
                Lset.units = 'umol photons m^-2 s^-1'
            if hisFME:
                Uset = hset.createVariable('ucm', 'f8',
                                           ('time', 'stations'))
                Uset.long_name = 'in-canopy flow'
                Uset.units = 'm s^-1'
            if hisTME:
                Tset = hset.createVariable('Tc', 'f8',
                                           ('time', 'stations'))
                Tset.long_name = 'coral temperature'
                Tset.units = 'K'

                Tloset = hset.createVariable('Tlo', 'f8',
                                             ('time', 'stations'))
                Tloset.long_name = 'lower thermal limit'
                Tloset.units = 'K'

                Thiset = hset.createVariable('Thi', 'f8',
                                             ('time', 'stations'))
                Thiset.long_name = 'upper thermal limit'
                Thiset.units = 'K'
            if hisPD:
                PSset = hset.createVariable('PS', 'f8',
                                            ('time', 'stations'))
                PSset.long_name = 'photosynthesis'
                PSset.units = '-'
            if hisPS:
                PTset = hset.createVariable('PT', 'f8',
                                            ('time', 'stations'))
                PTset.long_name = 'total population'
                PTset.units = '-'

                PHset = hset.createVariable('PH', 'f8',
                                            ('time', 'stations'))
                PHset.long_name = 'healthy population'
                PHset.units = '-'

                PRset = hset.createVariable('PR', 'f8',
                                            ('time', 'stations'))
                PRset.long_name = 'recovering population'
                PRset.units = '-'

                PPset = hset.createVariable('PP', 'f8',
                                            ('time', 'stations'))
                PPset.long_name = 'pale population'
                PPset.units = '-'

                PBset = hset.createVariable('PB', 'f8',
                                            ('time', 'stations'))
                PBset.long_name = 'bleached population'
                PBset.units = '-'
            if hisC:
                Gset = hset.createVariable('G', 'f8',
                                           ('time', 'stations'))
                Gset.long_name = 'calcification'
                Gset.units = 'kg m^-2 d^-1'
            if hisMD:
                DCset = hset.createVariable('dc', 'f8',
                                            ('time', 'stations'))
                DCset.long_name = 'plate diameter'
                DCset.units = 'm'

                HCset = hset.createVariable('hc', 'f8',
                                            ('time', 'stations'))
                HCset.long_name = 'coral height'
                HCset.units = 'm'

                BCset = hset.createVariable('bc', 'f8',
                                            ('time', 'stations'))
                BCset.long_name = 'base diameter'
                BCset.units = '-'

                TCset = hset.createVariable('tc', 'f8',
                                            ('time', 'stations'))
                TCset.long_name = 'plate thickness'
                TCset.units = 'm'

                ACset = hset.createVariable('ac', 'f8',
                                            ('time', 'stations'))
                ACset.long_name = 'axial distance'
                ACset.units = 'm'
                
                VCset = hset.createVariable('Vc', 'f8',
                                            ('time', 'stations'))
                VCset.long_name = 'coral volume'
                VCset.units = 'm^3'

            # data indices
            xs = xyn['x'].values
            ys = xyn['y'].values
            idx = np.zeros(len(xs))
            for s in range(len(xs)):
                idx[s] = ((xzw - xs[s]) ** 2 + (yzw - ys[s]) ** 2).argmin()
            idx = idx.astype(int)

            # data
            idates = dates.reset_index(drop=True)
            t[:] = (idates - date0).dt.days.values
            x[:] = xs
            y[:] = ys
            if hisLME:
                Lset[:, :] = coral.light[idx, :].transpose()
            if hisFME:
                Uset[:, :] = np.tile(coral.ucm, (len(idates), 1))[:, idx]
            if hisTME:
                Tset[:, :] = coral.temp[idx, :].transpose()
                if processes.tme:
                    Tloset[:, :] = np.tile(
                        coral.Tlo, (len(idates), 1))[:, idx]
                    Thiset[:, :] = np.tile(
                        coral.Thi, (len(idates), 1))[:, idx]
                else:
                    Tloset[:, :] = coral.Tlo * np.ones(
                        (len(idates), len(idx)))
                    Thiset[:, :] = coral.Thi * np.ones(
                        (len(idates), len(idx)))
            if hisPD:
                PSset[:, :] = coral.photo_rate[idx, :].transpose()
            if hisPS:
                PTset[:, :] = coral.pop_states[idx, :, :].sum(
                    axis=2).transpose()
                PHset[:, :] = coral.pop_states[idx, :, 0].transpose()
                PRset[:, :] = coral.pop_states[idx, :, 1].transpose()
                PPset[:, :] = coral.pop_states[idx, :, 2].transpose()
                PBset[:, :] = coral.pop_states[idx, :, 3].transpose()
            if hisC:
                Gset[:, :] = coral.calc[idx, :].transpose()
            if hisMD:
                DCset[:, :] = np.tile(coral.dc, (len(idates), 1))[:, idx]
                HCset[:, :] = np.tile(coral.hc, (len(idates), 1))[:, idx]
                BCset[:, :] = np.tile(coral.bc, (len(idates), 1))[:, idx]
                TCset[:, :] = np.tile(coral.tc, (len(idates), 1))[:, idx]
                ACset[:, :] = np.tile(coral.ac, (len(idates), 1))[:, idx]
        else:
            hset = Dataset(hisfilef, mode='a')
            # data indices
            xs = xyn['x'].values
            ys = xyn['y'].values
            idx = np.zeros(len(xs))
            for s in range(len(xs)):
                idx[s] = ((xzw - xs[s]) ** 2 + (yzw - ys[s]) ** 2).argmin()
            idx = idx.astype(int)
            # date conversion
            idates = dates.reset_index(drop=True)
            t = (idates - date0).dt.days.values
            # append data
            hset['time'][:] = np.append(hset['time'][:], t)
            if hisLME:
                hset['Iz'][t, :] = coral.light[idx, :].transpose()
            if hisFME:
                hset['ucm'][t, :] = np.tile(
                    coral.ucm, (len(idates), 1))[:, idx]
            if hisTME:
                hset['Tc'][t, :] = coral.temp[idx, :].transpose()
                if processes.tme:
                    hset['Tlo'][t, :] = np.tile(
                            coral.Tlo, (len(idates), 1))[:, idx]
                    hset['Thi'][t, :] = np.tile(
                            coral.Thi, (len(idates), 1))[:, idx]
                else:
                    hset['Tlo'][t, :] = coral.Tlo * np.ones(
                            (len(idates), len(idx)))
                    hset['Thi'][t, :] = coral.Thi * np.ones(
                            (len(idates), len(idx)))
            if hisPD:
                hset['PS'][t, :] = coral.photo_rate[idx, :].transpose()
            if hisPS:
                hset['PT'][t, :] = coral.pop_states[idx, :, :].sum(
                    axis=2).transpose()
                hset['PH'][t, :] = coral.pop_states[idx, :, 0].transpose()
                hset['PR'][t, :] = coral.pop_states[idx, :, 1].transpose()
                hset['PP'][t, :] = coral.pop_states[idx, :, 2].transpose()
                hset['PB'][t, :] = coral.pop_states[idx, :, 3].transpose()
            if hisC:
                hset['G'][t, :] = coral.calc[idx, :].transpose()
            if hisMD:
                hset['dc'][t, :] = np.tile(
                    coral.dc, (len(idates), 1))[:, idx]
                hset['hc'][t, :] = np.tile(
                    coral.hc, (len(idates), 1))[:, idx]
                hset['bc'][t, :] = np.tile(
                    coral.bc, (len(idates), 1))[:, idx]
                hset['tc'][t, :] = np.tile(
                    coral.tc, (len(idates), 1))[:, idx]
                hset['ac'][t, :] = np.tile(
                    coral.ac, (len(idates), 1))[:, idx]
        hset.close()

# =============================================================================
# # # # environmental time-series
# =============================================================================
env = Environment()
env.fromFile('light', fLight, inputdir)
try:
    env.fromFile('LAC', fLAC, inputdir)
except FileNotFoundError:
    env.Kd = pd.DataFrame(index=env.light.index,
                                  data={'lac': constants.Kd0})
env.fromFile('temperature', fTemperature, inputdir)
try:
    env.fromFile('acidity', fAcidity, inputdir)
except FileNotFoundError:
    env.acid = pd.DataFrame(index=env.light.index,
                                    data={'arag': constants.omegaA0})
env.fromFile('storm', fStorm, inputdir)

print('Environmental conditions defined.\n')

# =============================================================================
# # # # coral definition
# =============================================================================
coral = Morphology(dc0 * np.ones(ndxi),
                   hc0 * np.ones(ndxi),
                   bc0 * np.ones(ndxi),
                   tc0 * np.ones(ndxi),
                   ac0 * np.ones(ndxi))
coral.Csp = 1.
coral.K = K
coral.P0 = P0

# =============================================================================
# # # # run the model
# =============================================================================
print('Start time : {}\n'
      .format(datetime.datetime.now()))

with tqdm(range(int(Y))) as pbar:
    for i in pbar:
        if processes.pfd or i == 0:
            # # update hydrodynamic model
            pbar.set_postfix(inner_loop='update Delft3D')
            modelDIMR.update(mtpervt)
            
            # # extract variables from DFM via BMI
            pbar.set_postfix(inner_loop='extract variables')
            # flow characteristics
            is_sumvalsnd = modelFM.get_var('is_sumvalsnd')
            is_maxvalsnd = modelFM.get_var('is_maxvalsnd')
            Uwave = modelFM.get_var('Uorb')[range(ndxi)]
            Twave = modelFM.get_var('twav')[range(ndxi)]
            # param[range(ndxi), i]
            # > i = 0 : shear stress   [tau]
            # > i = 1 : flow velocity  [vel]
            # > i = 2 : water depth    [wd]
            # morphological characteristics
            diaveg = modelFM.get_var('diaveg')
            dcmodel = diaveg[range(ndxi)]
            stemheight = modelFM.get_var('stemheight')
            rnveg = modelFM.get_var('rnveg')
            
            # # calculate (mean) values from DFM data
            Ucurr = is_sumvalsnd[range(ndxi), 1] / mtpervt
            depth = is_sumvalsnd[range(ndxi), 2] / mtpervt
        
        # # dimensions
        spacetime = np.array(
            [ndxi, len(env.dates[env.dates.dt.year == years[i]])])
        
        # depth = 10. * np.ones(ndxi)
        # Ucurr = .5 * np.ones(ndxi)
        # Uwave = .5 * np.ones(ndxi)
        # Twave = 4. * np.ones(ndxi)
        
        # # environment
        pbar.set_postfix(inner_loop='coral environment')
        # light micro-environment
        lme = Light(
            env.light[env.light.index.year == years[i]].values[:, 0],
            env.Kd[env.Kd.index.year == years[i]].values[:, 0],
            depth)
        lme.repLight(coral)
        # flow micro-environment
        fme = Flow(Ucurr, Uwave, depth, Twave)
        fme.waveCurrent(coral, incanopy=processes.fme)
        fme.TBL(coral)
        # thermal micro-environment
        tme = Temperature(
            env.tempK[env.tempK.index.year == years[i]].values[:, 0])
        tme.coralTemperature(coral)
        
        # # physiology
        pbar.set_postfix(inner_loop='coral physiology')
        # photosynthetic dependencies
        if i == 0:
            phd = Photosynthesis(
                env.light[env.light.index.year == years[i]].values[:, 0],
                firstYear=True)
        else:
            phd = Photosynthesis(
                env.light[env.light.index.year == years[i]].values[:, 0],
                firstYear=False)
        phd.photosyn(coral, env, years[i])
        # population states
        ps = PopStates()
        ps.popstates_t(coral)
        # calcification
        cr = Calcification()
        cr.calRate(
            coral, env.acid[env.acid.index.year == years[i]].values[:, 0])
        
        # # morphology
        pbar.set_postfix(inner_loop='coral morphology')
        # morphological development
        mor = MorDevelopment(
            coral.calc.sum(axis=1), depth,
            env.Kd[env.Kd.index.year == years[i]].values[:, 0])
        mor.update(
            coral, env.light[env.light.index.year == years[i]].values[:, 0])
        
        # # storm damage
        if env.stormcat[env.stormcat.index == years[i]].values > 0:
            # return coral data to hydrodynamic model
            pbar.set_postfix(inner_loop='storm - update morphology in Delft3D')
            # translate model parameters
            rnveg = coral.morph2vegden()
            diaveg = coral.dcRep
            stemheight = coral.hc
            # reset counters
            is_sumvalsnd.fill(0.)
            is_maxvalsnd.fill(0.)
            # push counters and updated coral field to model
            modelFM.set_var('is_sumvalsnd', is_sumvalsnd)
            modelFM.set_var('is_maxvalsnd', is_maxvalsnd)
            modelFM.set_var('rnveg', rnveg)
            modelFM.set_var('diaveg', diaveg)
            modelFM.set_var('stemheight', stemheight)
            
            # run storm conditions
            pbar.set_postfix(inner_loop='storm - update Delft3D')
            modelDIMR.update(mtpervt_storm)
            
            # extract variables from DFM via BMI
            pbar.set_postfix(inner_loop='storm - extract variables')
            is_sumvalsnd = modelFM.get_var('is_sumvalsnd')
            is_maxvalsnd = modelFM.get_var('is_maxvalsnd')
            Uwave = modelFM.get_var('Uorb')[range(ndxi)]
            rnveg = modelFM.get_var('rnveg')
            diaveg = modelFM.get_var('diaveg')
            stemheight = modelFM.get_var('stemheight')
            # maximum flow velocity
            Ucurr = is_maxvalsnd[range(ndxi), 1]
            # Uwave = .8
            # Ucurr = 1.
            
            # storm flow environment
            pbar.set_postfix(inner_loop='storm - dislodgement')
            sfe = Flow(Ucurr, Uwave, None, None)
            sfe.waveCurrent(coral, incanopy=False)
            # storm dislodgement criterion
            sdc = Dislodgement()
            sdc.update(coral)

        # # recruitment / spawning
        pbar.set_postfix(inner_loop='coral recruitment')
        rec = Recruitment()
        rec.update(coral)
        
        if processes.pfd:
            # # return coral data to hydrodynamic model
            pbar.set_postfix(inner_loop='update morphology in Delft3D')
            # translate model parameters
            rnveg = coral.morph2vegden()
            diaveg = coral.dcRep
            stemheight = coral.hc
            # reset counters
            is_sumvalsnd.fill(0.)
            is_maxvalsnd.fill(0.)
            # push counters and updated coral field to model
            modelFM.set_var('is_sumvalsnd', is_sumvalsnd)
            modelFM.set_var('is_maxvalsnd', is_maxvalsnd)
            modelFM.set_var('rnveg', rnveg)
            modelFM.set_var('diaveg', diaveg)
            modelFM.set_var('stemheight', stemheight)
            
        # # export model results
        pbar.set_postfix(inner_loop='write output file(s)')
        if i == 0:
            # map-file
            outputMap(
                coral,
                L2mfile, U2mfile, T2mfile, PS2mfile, P2mfile, G2mfile, M2mfile,
                years[i], True, mapfilef)
            # his-file
            outputHis(
                coral,
                L2hfile, U2hfile, T2hfile, PS2hfile, P2hfile, G2hfile, M2hfile,
                env.dates[env.dates.dt.year == years[i]], True, hisfilef)
        else:
            # map-file
            outputMap(
                coral,
                L2mfile, U2mfile, T2mfile, PS2mfile, P2mfile, G2mfile, M2mfile,
                years[i], False, mapfilef)
            # his-file
            outputHis(
                coral,
                L2hfile, U2hfile, T2hfile, PS2hfile, P2hfile, G2hfile, M2hfile,
                env.dates[env.dates.dt.year == years[i]], False, hisfilef)

# =============================================================================
# # # # finalize the model
# =============================================================================
modelDIMR.finalize()
print('\nModel finalized.')
print('\nEnd time   : {0}'.format(datetime.datetime.now().time()))
