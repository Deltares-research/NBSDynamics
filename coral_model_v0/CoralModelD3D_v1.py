# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:09:07 2019

@author: Gijs Hendrickx
"""

# =============================================================================
# # # # import packages
# =============================================================================
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
model_folder = os.path.join('MiniModel')

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
# morphology > { Lc } > { dc, hc, bc, tc, ac }
M2mfile = True

# # map-file
# map-file directory
mapfile = 'CoralModel_map.nc'
mapfilef = os.path.join(outputfolder, mapfile)
# time-interval > annually

# # # history > time-series
# # data to output file
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
# morphology > { Lc } > { dc, hc, bc, tc, ac }
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
xvbndmax = xbndmax  # (xbndmin + xbndmax) / 2
yvbndmin = ybndmin
yvbndmax = 700.

# =============================================================================
# # # # enabling/disabling processes
# =============================================================================
# in-canopy flow
cft = True
# thermal boundary layer
tbl = False

# # process dependencies
if not cft:
    tbl = False

# # corfac
corfac = 1.

# =============================================================================
# # # # model constants
# =============================================================================
# species constant
Csp = 1.

# # light
# light-attenuation coefficient
Kd0 = .1
# maximum saturation intensity (umol photons m^-2 s^-1)
Ikmax = 400.

# # hydrodynamics
# Smagorinsky constant
Cs = .17
# inertia constant
Cm = 1.7
# friction coefficient
Cf = .01
# wave-current angle (degrees)
wcangle = 0.

# # temperature
# TBL coefficient
K0 = 80.
# thermal acclimation coefficient
Kvar = 2.45

# # acidity
# aragonite saturation state
omega0 = 5.

# # morphology
# overall form proportionality constant
Xf = .1
# flow plate proportionality constant
Xpu = .1
# light spacing proportionality constant
XsI = .1
# flow spacing proportionality constant
Xsu = .1

# # dislodgement
# tensile strength substratum [N m^-2]
sigmat = 2e5

# # recruitment
# probability of settlement [-]
ps = 1e-4
# larvae due to spawning [-]
Nl = 1e6
# larval diameter [m]
dl = 1e-3

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

# =============================================================================
# # # # read time-series
# =============================================================================
# # light conditions > { I0['par'] }
filef = os.path.join(inputdir, 'TS_PAR.txt')
I0 = pd.read_csv(filef, sep='\t')
I0['date'] = pd.to_datetime(I0['date'])
I0.set_index('date', inplace=True)

filef = os.path.join(inputdir, 'TS_Kd.txt')
if os.path.exists(filef):
    Kd = pd.read_csv(filef, sep='\t')
    Kd['date'] = pd.to_datetime(Kd['date'])
    Kd.set_index('date', inplace=True)

# # thermal conditions > { T['sst'] }
filef = os.path.join(inputdir, 'TS_SST.txt')
T = pd.read_csv(filef, sep='\t')
T['date'] = pd.to_datetime(T['date'])
T.set_index('date', inplace=True)

# # aragonite conditions > { omega['arg'] }
filef = os.path.join(inputdir, 'TS_ARG.txt')
if os.path.exists(filef):
    omega = pd.read_csv(filef, sep='\t')
    omega['date'] = pd.to_datetime(omega['date'])
    omega.set_index('date', inplace=True)

# # storm categories > { H['stormcat'] }
filef = os.path.join(inputdir, 'TS_stormcat.txt')
H = pd.read_csv(filef, sep='\t')
H.set_index('year', inplace=True)

# # dates
dates = T.reset_index().drop('sst', axis=1)
dates = pd.to_datetime(dates['date'])

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
# # # # definition of basic functions for coral development
# =============================================================================
# # light


def baseheight_light(dc, hc, bc, tc, h, Kd, thetamax=.5*np.pi):
    """
    The height of the base of the coral that receives light and so is not
    shaded by the plateau.

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
    h : numeric
        Water depth [m].
    Kd : numeric
        Light-attenuation coefficient [m^-1].
    thetamax : numeric, optional
        Maximum spreading of light at the top of the water column [radians].

    Returns
    -------
    L : numeric
        Height of the base that receives light [m].
    """
    # # functions

    def light_spreading(h, hc, tc, Kd):
        """
        The spreading of light in the water as function of the water depth and
        the coral morphology.
        """
        theta = thetamax * np.exp(- Kd * (h - hc + tc))

        return theta

    # # calculations
    # light spreading, theta
    theta = light_spreading(h, hc, tc, Kd)
    # base-height receiving light, L
    L = hc - tc - (dc - bc) / (2. * np.tan(.5 * theta))

    # # output
    return L


def biomass(dc, hc, bc, tc, h, Kd, thetamax=.5*np.pi):
    """
    The biomass of the coral expressed as coral surface area that receives
    light and so can contribute to the photosynthesis.

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
    h : numeric
        Water depth [m].
    Kd : numeric
        Light-attenuation coefficient [m^-1].
    thetamax : numeric, optional
        Maximum spreading of light at the top of the water column [radians].

    Returns
    -------
    Bc : numeric
        Biomass of the coral [m^2].
    """
    # # calculations
    # base-height receiving light, L
    L = baseheight_light(dc, hc, bc, tc, h, Kd, thetamax=thetamax)
    try:
        if L < 0:
            L = 0.
    except ValueError:
        L[L < 0] = 0.
    # biomass
    Bc = np.pi * (.25 * dc ** 2 + dc * tc + bc * L)

    # # output
    return Bc


def av_light(I0, Kd, h, dc, hc, bc, tc, thetamax=.5*np.pi, th_unit='rad'):
    """
    The representative light-intensity that the coral catches and thereby is
    able to use for photosynthesis.

    Parameters
    ----------
    I0 : numeric
        Light-intensity at the surface water (but within the water column)
        [mol photons m^-2 s^-1].
    Kd : numeric
        Light-attenuation coefficient [m^-1].
    h : numeric
        Water depth [m].
    hc : numeric
        Height of the coral [m].
    dc : numeric
        Diameter of the upper part of the coral [m].
    bc : numeric
        Diameter of the base of the coral [m].
    thetamax : numeric, optional
        Maximum spreading of light at the top of the watercolumn [radians].
    th_unit : str, optional
        Unit of thetamax.

    Returns
    -------
    Iz_av : numeric, default
        Biomass-averaged light-intensity [mol photons m^-2 s^-1].
    """
    # # check input
    # unit options
    units = ['deg', 'rad']
    if th_unit not in units:
        raise ValueError(
                'Unit of thetamax is not in options: {0}'
                .format(units))
    # convert th_unit
    if th_unit == 'deg':
        thetamax = np.deg2rad(thetamax)

    # # calculations
    # coral biomass
    Bc = biomass(dc, hc, bc, tc, h, Kd, thetamax=thetamax)
    # base height receiving light
    L = baseheight_light(dc, hc, bc, tc, h, Kd, thetamax=thetamax)
    try:
        if L < 0.:
            L = 0.
    except ValueError:
        L[L < 0] = 0.
    # light-catchment top
    Iz_top = .25 * np.pi * dc ** 2 * I0 * np.exp(-Kd * (h - hc))
    # light-catchment side plateau
    Iz_plat = (np.pi * dc * I0) / Kd * (np.exp(-Kd * (h - hc)) -
                                        np.exp(-Kd * (h - hc + tc)))
    # light-catchment side base
    Iz_base = (np.pi * bc * I0) / Kd * (np.exp(-Kd * (h - L)) -
                                        np.exp(-Kd * h))
    # total light-catchment
    Iztot = Iz_top + Iz_plat + Iz_base
    # biomass-averaged light-intensity
    try:
        if Bc == 0.:
            Iz_av = I0 * np.exp(-Kd * h)
        else:
            Iz_av = Iztot / Bc
    except ValueError:
        Iz_av = I0 * np.exp(-Kd * h)
        Iz_av[Bc > 0] = Iztot[Bc > 0] / Bc[Bc > 0]

    # # output
    return Iz_av


# # hydrodynamics


def wavecurrent(uwav, ucur, wcangle, alphaw=1., alphac=1., angle_unit='deg'):
    """
    Resulting flow velocity due to wve-current interactions.

    Parameters
    ----------
    uwav : numeric
        Wave orbital velocity [m s^-1].
    ucur : numeric
        Current velocity [m s^-1].
    wcangle : numeric
        Angle between wave and current velocity vectors [deg / rad].
    alphaw : numeric, optional
        Wave-attenuation coefficient, waves [-].
    alphac : numeric, optional
        Wave-attenuation coefficient, current [-]

    Returns
    -------
    um : numeric
        Mean flow velocity due to wave-current interactions [m s^-1]..
    """
    # # check input
    # unit check
    units = ['deg', 'rad']
    if angle_unit not in units:
        raise ValueError(
                'Specified unit of angle not in options: {0}.'
                .format(units))
    # translate angle to radians
    if angle_unit == 'deg':
        wcangle = np.deg2rad(wcangle)

    # # calculations
    um = np.sqrt((alphaw * uwav) ** 2 + (alphac * ucur) ** 2 +
                 (2. * alphaw * uwav * alphac * ucur * np.cos(wcangle)))

    # # output
    return um


def wac(uf, T, lp, Ls, Ld, Cm, maxiter=1e3, err=1e-3,
        allow_simp=False, h=None, dc=None, hc=None, ac=None):
    """
    The wave-attenuation coefficient as function of the wave-characteristics
    and the geometry of the canopy. The above-canopy (or free) flow is assumed
    to be known, and thus used as input. Note that this is different from the
    bulk flow.

    Solving method is according to the complex representation of the wave.

    Solving for a multi-layer canopy is possible. If so, layer 0 is the above-
    canopy layer, layer 1 the top layer of the canopy, etc. The shear at the
    bottom is not taken into account. Thus the arrays of [lp], [Ls] and [Ld]
    must have the same length.

    --------------------------------------- MSL


                                            layer 0


    ################# - - - - ----- - - - - -
    #################         dh1           layer 1
    ################# - - - - ----- - - - - -
         ######
         ######
         ######               dh2           layer 2
         ######
         ######
    ======================================= BL
    =======================================
    =======================================

    Parameters
    ----------
    uf : numeric
        Free flow velocity [m s^-1].
    T : numeric
        Wave period [s].
    lp : numeric, array
        Planar lambda-parameter [-].
    Ls : numeric, array
        Shear length-scale [m].
    Ld : numeric, array
        Drag length-scale [m].
    Cm : numeric
        Inertia coefficient [-].
    maxiter : numeric, integer, optional
        Maximum number of iterations for the Newton-Raphson method, and the
        iteration needed for the multi-layer canopy.
    err : numeric, optional
        Maximum allowable absolute error in case of a multi-layer canopy.
    allow_simp : boolean, optional
        When the maximum number of iterations for the Newton-Raphson method is
        reached, allow for simplified expressions. Enabling this option
        requires the definition of the (1) water depth; (2) coral height;
        (3) coral (representative) diameter; and (4) axial distance.
    h : None, numeric, optional
        Water depth [m].
    hc : None, numeric, optional
        Coral height [m].
    dc : None, numeric, optional
        Coral (representative) diameter [m].
    ac : None, numeric, optional
        Axial distance [m].

    Returns
    -------
    aw : numeric, array
        Wave-attenuation coefficient [-].
    """
    # # check input
    # integer > float
    if isinstance(lp, int):
        lp = float(lp)
    if isinstance(Ls, int):
        Ls = float(Ls)
    if isinstance(Ld, int):
        Ld = float(Ld)
    # correspondance in type and length
    if not type(lp) == type(Ls) == type(Ld):
        raise ValueError(
                '[lp], [Ls] and [Ld] must be of the same type.')
    if not isinstance(lp, float):
        if not len(lp) == len(Ls) == len(Ld):
            raise ValueError(
                    '[lp], [Ls] and [Ld] must have same lengths.')
    # required input for simplified approach
    if allow_simp:
        if h is None or dc is None or hc is None or ac is None:
            raise ValueError(
                    'Enabling the option of the simplified approximation if '
                    'needed requires the definition of the (1) water depth; '
                    '(2) coral height; (3) coral (representative) diameter; '
                    'and (4) axial distance. Type wac? for help.')

    # # define functions

    def func(beta, beta0, beta1, af, lp, Ls, Ld, Cm):
        """
        Solution of the complex wave representation for the {k}-th layer.

        When [beta_] is set to [None], its influence via the shear component is
        not taken into account.

        Parameters (new)
        ----------
        beta : numeric, complex
            Copmlex wave-attenuation coefficient of layer {k} [-].
        beta0 : numeric, complex
            Complex wave-attenuation coefficient of layer {k - 1} [-].
        beta1 : numeric, complex
            Complex wave-attenuation coefficient of layer {k + 1} [-].

        Returns
        -------
        f : function, complex
            Complex function where the solution is found by setting it equal to
            zero [-].
        """
        # # check input
        # convert NaN > None
        if beta0 is not None:
            if np.isnan(beta0):
                beta0 = None
        if beta1 is not None:
            if np.isnan(beta1):
                beta1 = None
        # exclude layers from calculations
        if beta0 is None:
            beta0 = beta
        if beta1 is None:
            beta1 = beta

        # # calculations
        f = 1j * (beta - 1.) + (8. * af) / (3. * np.pi) * (
                - (abs(beta0 - beta) * (beta0 - beta)) / Ls
                + (abs(beta - beta1) * (beta - beta1)) / Ls
                + (abs(beta) * beta) / Ld) + 1j * beta * ((Cm * lp) /
                                                          (1. - lp))

        # # output
        return f

    def deriv(beta, beta0, beta1, af, lp, Ls, Ld, Cm):
        """
        The derivative of [func(beta)] to beta to find the root of this
        function.

        When [beta_] is set to [None], its influence via the shear component is
        not taken into account.

        Parameters (new)
        ----------
        beta : numeric, complex
            Copmlex wave-attenuation coefficient of layer {k} [-].
        beta0 : numeric, complex
            Complex wave-attenuation coefficient of layer {k - 1} [-].
        beta1 : numeric, complex
            Complex wave-attenuation coefficient of layer {k + 1} [-].

        Returns
        -------
        df : function, complex
            Derivative of the omplex function [func(beta)], where the solution
            is found by setting it equal to zero [-].
        """
        # # check input
        # convert NaN > None
        if beta0 is not None:
            if np.isnan(beta0):
                beta0 = None
        if beta1 is not None:
            if np.isnan(beta1):
                beta1 = None
        # exclude layers from calculations
        if beta0 is None:
            beta0 = beta
        if beta1 is None:
            beta1 = beta

        # # calculations
        # piecewise components
        if beta0 == beta:
            S0 = 0.
        else:
            S0 = (beta0 - beta) ** 2 / abs(beta0 - beta) - abs(beta0 - beta)

        if beta1 == beta:
            S1 = 0.
        else:
            S1 = (beta - beta1) ** 2 / abs(beta - beta1) + abs(beta - beta1)

        if beta == 0.:
            D = 0.
        else:
            D = beta ** 2 / abs(beta) + abs(beta)

        # derivate
        df = 1j + (8. * af) / (3. * np.pi) * (
                - S0 / Ls + S1 / Ls + D / Ld) + 1j * (Cm * lp) / (1. - lp)

        # # output
        return df

    # # assign correct values to working arrays
    if not isinstance(lp, float):
        lp = np.append(None, lp)
        Ls = np.append(None, Ls)
        Ls = np.append(Ls, None)
        Ld = np.append(None, Ld)

    # # calculations
    af = (uf * T) / (2. * np.pi)
    # one-layer canopy
    if isinstance(lp, float):
        args = [1., None, af, lp, Ls, Ld, Cm]
        try:
            aw = abs(newton(func, x0=complex(.1, .1), fprime=deriv, args=args,
                            maxiter=int(maxiter)))
        except RuntimeError:
            X = Ld / Ls * (hc / (h - hc) + 1.)
            aw = (X - np.sqrt(X)) / (X - 1.)
    # multi-layer canopy
    if not isinstance(lp, float):
        beta = np.linspace(0., 1., len(lp) + 1, dtype=complex)
        beta[0] = 1.
        beta[-1] = None
        betai = beta
        for i in range(int(maxiter)):
            # convergance step
            for k in range(1, len(lp)):
                args = [beta[k-1], beta[k+1], af, lp[k],
                        Ls[k], Ld[k], Cm]
                try:
                    betai[k] = newton(func, x0=complex(.5, .5), fprime=deriv,
                                      args=args, maxiter=int(maxiter))
                except RuntimeError:
                    print('WARNING: RuntimeError ignored, latest output used.')
                    break
            if all(abs(e) < err for e in
                   abs(beta[1:-1] - betai[1:-1]) / abs(beta[1:-1])):
                aw = abs(betai[1:-1])
                break
            else:
                beta[1:-1] = betai[1:-1]
            if i == maxiter:
                aw = abs(beta[1:-1])
                print('WARNING: '
                      'maximum iterations reached before convergence.')

    # # output
    return aw


def drag(h, ub, T, dc, hc, ac, Cs, Cm,
         e=1e-7, psi=2., theta=.5, maxiter_k=1e5,
         method='complex', maxiter_aw=1e3,
         print_iter=False, output='Cd'):
    """
    Iteratively determining the drag coefficient as function of flux of water
    and the coral morphology in which corals are represented as (multi-layer)
    cylinders as displayed below (frontal view).

    --------------------------------------- MSL


                                            layer 0


    ################# - - - - ----- - - - - -
    #################         dh1           layer 1
    ################# - - - - ----- - - - - -
         ######
         ######
         ######               dh2           layer 2
         ######
         ######
    ======================================= BL
    =======================================
    =======================================

    The corals are placed in a staggered manner as displayed below
    (plan view).

    O _ _ _ _ _ O     --
    _ _ _ _ _ _ _
    _ _ _ _ _ _ _
    _ _ _ O _ _ _     ac
    _ _ _ _ _ _ _
    _ _ _ _ _ _ _
    O _ _ _ _ _ O     --

    |     ac    |

    NOTE: Definition of the wave-attenuation coefficient outside this function.

    Parameters
    ----------
    h : numeric
        Water depth [m].
    ub : numeric
        Bulk flow velocity [m s^-1].
    T : numeric
        Wave period [s].
    dc : numeric, array
        Coral (or cylinder) diameter [m].
        If array, diameter per canopy layer [m].
    hc : numeric, array
        Coral (or canopy) height [m].
        If array, height per canopy layer [m].
    ac : numeric, array
        Axial distance between corals (or cylinders) [m].
        If array, axial distance per canopy layer [m].
    Cs : numeric
        Friction coefficient [-].
    Cm : numeric
        Inertia coefficient [-].
    e : numeric, optional
        Maximum allowable relative error.
    psi : numeric, optional
        Ratio of lateral over streamwise spacing of corals (or cylinders) [-].
    theta : numeric, optional
        Update ratio for the above-canopy flow [-].
    maxiter_k : integer, numeric, optional
        Maximum number of iterations for the wave-attenuation coefficient taken
        over the canopy layers [-].
    method : string, optional
        The method to determine the wave-attenuation coefficient.
        Options: [complex (default), simplified]
    maxiter_aw : integer, numeric, optional
        Maximum number of iterations for the Newton-Raphson method for
        determining the wave-attenaution coefficient according the [complex]
        method.
    output : string, optional
        Definition of output, options:
            ['Cd']  : drag coefficient
            ['aw']  : wave-attenuation coefficient
            ['uc']  : constricted in-canopy flow
            ['all'] : drag coefficient, wave-attenuation coefficient, and
                      constricted in-canopy flow

    Returns
    -------
    Cd : numeric
        Drag coefficient [-].
    aw : numeric
        Wave-attenuation coefficient [-].
    uc : numeric
        Constricted in-canopy flow [m s^-1].
    """
    # # check input
    # integers to floats for input check
    if isinstance(hc, int):
        hc = float(hc)
    if isinstance(dc, int):
        dc = float(dc)
    if isinstance(ac, int):
        ac = float(ac)
    # morphology
    if not type(hc) == type(dc):
        raise ValueError(
                'Types of [hc] and [dc] must be the same.')
    if isinstance(hc, float) and isinstance(dc, float):
        if not isinstance(ac, float):
            raise ValueError(
                    '[ac] must be float if [hc] and [dc] are floats.')
    elif isinstance(hc, np.ndarray) and isinstance(dc, np.ndarray):
        if not len(hc) == len(dc):
            raise ValueError(
                    '[hc] and [dc] must have same length.')
        if isinstance(ac, np.ndarray):
            if len(ac) > 1:
                if not len(ac) == len(dc):
                    raise ValueError(
                            'Length of [ac] must be 1 or equal to the length '
                            'of [hc] and [dc]. (Or [ac] is float)')

    # wave-attenuation coefficient method
    methods = ['complex', 'simplified']
    if method not in methods:
        raise ValueError('Invalid method. Expected one of: {0}'
                         .format(methods))
    if not isinstance(hc, float) and isinstance(dc, float):
        if not method == 'complex':
            method = 'complex'
            print('WARNING: '
                  'Multi-layer canopy flow only suppported by complex method.')
    # output definition
    outputs = ['Cd', 'aw', 'uc', 'all']
    if output not in outputs:
        raise ValueError('Invalid output definition. Expected one of: {0}'
                         .format(outputs))

    # # calculations
    # constants
    nu = 1e-6

    # geometric parameters
    Ap = .25 * np.pi * dc ** 2
    Af = hc * dc
    AT = .5 * ac ** 2
    lp = Ap / AT
    lf = Af / AT
    Ls = hc / (Cs ** 2)

    if isinstance(hc, float) and isinstance(dc, float):
        # # emergent vs. submerged
        if h <= hc:
            # emergent
            up = ub / (1. - lp)
            uc = ((1. - lp) / (1. - np.sqrt((4. * lp) / (psi * np.pi)))) * up
            Re = (uc * dc) / nu
            Cd = 1. + 10. * Re ** (- 2. / 3.)
            aw = 1.
        else:
            # submerged
            # initial values before iteration
            uf = ub
            Cd = 1.

            # iteration
            for k in range(int(maxiter_k)):
                Ld = (2. * hc * (1. - lp)) / (Cd * lf)

                # wave-attenuation
                if hc == 0.:
                    aw = 1.
                elif method == 'complex':
                    aw = wac(uf, T, lp, Ls, Ld, Cm, maxiter=maxiter_aw,
                             allow_simp=True, h=h, dc=dc, hc=hc, ac=ac)
                else:
                    omega = (2. * np.pi) / T
                    af = uf / omega
                    if af / (ac - dc) < 1.:
                        aw = (1. - lp) / (1. + (Cm - 1.) * lp)
                        print('Inertia dominated domain')
                    elif af / (ac - dc) > 100.:
                        aw = np.sqrt(Ld / Ls)
                        print('Unidirectional domain')
                    else:
                        aw = None
                        print('General domain - no WAC defined.')

                up = aw * uf
                uc = (1. - lp) / (1. - np.sqrt((4. * lp) / (psi * np.pi))) * up
                Re = (uc * dc) / nu
                Cdk = 1. + 10. * Re ** (- 2. / 3.)
                if abs((Cdk - Cd) / Cdk) <= e:
                    break
                else:
                    Cd = float(Cdk)
                    uf = abs((1. - theta) * uf +
                             theta * (h * ub - hc * up) / (h - hc))

                # maximum iterations
                if k == maxiter_k:
                    print('WARNING: Maximum number of iterations reached: {0}'
                          .format(maxiter_k))

            if up > uf:
                print('WARNING: In-canopy flow larger than above-canopy flow. '
                      'Physically not sound!')
                print('hc = {0}\ndc = {1}\nac = {2}'.format(hc, dc, ac))
            if print_iter:
                print('Number of iterations for drag coefficient: {0}'
                      .format(k + 1))
    elif isinstance(hc, np.ndarray) and isinstance(dc, np.ndarray):
        # # emergent vs. submerged
        if h <= hc.sum():
            # emergent
            raise ValueError(
                    'Multi-layer emergent canopy calculations not included.')
        else:
            # submerged
            # initial values before iteration
            uf = ub
            Cd = np.ones(len(hc))

            # iteration
            for k in range(int(maxiter_k)):
                Ld = (2. * hc * (1. - lp)) / (Cd * lf)

                # wave-attenuation
                aw = wac(uf, T, lp, Ls, Ld, Cm, maxiter=maxiter_aw)

                # updated drag coefficient
                up = aw * uf
                uc = (1. - lp) / (1. - np.sqrt((4. * lp) / (psi * np.pi))) * up
                Re = (uc * dc) / nu
                Cdk = 1. + 10. * Re ** (- 2. / 3.)
                if all(np.abs(Cdk - Cd) / Cdk <= e):
                    break
                else:
                    Cd = Cdk
                    uf = abs((1. - theta) * uf +
                             theta * (h * ub - np.sum(hc * up)) /
                             (h - hc.sum()))

                # maximum iterations
                if k == maxiter_k:
                    print('WARNING: Maximum number of iterations reached: {0}'
                          .format(maxiter_k))

            if any(up > uf):
                print('WARNING: In-canopy flow larger than above-canopy flow. '
                      'Physically not sound!')
                print('dc = {0}\nhc = {1}\nac = {2}'.format(dc, hc, ac))
            if print_iter:
                print('Number of iterations for drag coefficient: {0}'
                      .format(k + 1))

    # # output
    if output == 'Cd':
        out = Cd
    elif output == 'aw':
        out = aw
    elif output == 'uc':
        out = uc
    elif output == 'all':
        out = np.array([Cd, aw, uc])

    return out


def VBL(uc, Cf, rd=500, nu=1e-6):
    """
    The thickness of the velocity boundary layer around a cylinder loaded by an
    unidirectional flow.

    Parameters
    ----------
    uc : numeric
        Constricted in-canopy flow [m s^-1].
    Cf : numeric
        Friction coefficient [-].
    rd : numeric, optional
        Velocity boundary layer wall-coordinate.
    nu : numeric, optional
        Kinematic viscosity [m^2 s^-1].

    Returns
    -------
    delta : numeric
        Thickness of the velocity boundary layer [m].
    """
    # # calculations
    try:
        if uc > 0:
            delta = ((rd * nu) / (np.sqrt(Cf) * uc))
        else:
            delta = 0.
    except ValueError:
        delta = np.zeros(uc.shape)
        delta[uc > 0] = ((rd * nu) / (np.sqrt(Cf) * uc[uc > 0]))

    # # output
    return delta


def TBL(uc, Cf, rd=500, nu=1e-6, alpha=1e-7):
    """
    The thickness of the thermal boundary layer around a cylinder loaded by an
    unidirectional flow.

    Parameters
    ----------
    uc : numeric
        Constricted in-canopy flow [m s^-1].
    Cf : numeric
        Friction coefficient [-].
    rd : numeric, optional
        Velocity boundary layer wall-coordinate.
    nu : numeric, optional
        Kinematic viscosity [m^2 s^-1].
    alpha : numeric, optional
        Thermal diffusivity [m^2 s^-1].

    Returns
    -------
    deltat : numeric
        Thickness of the thermal boundary layer [m].
    """
    delta = VBL(uc, Cf, rd=rd, nu=nu)
    deltat = delta * ((alpha / nu) ** (1 / 3))

    return deltat


# # coral temperature

def delta_Tc(uc, Iz, Cf, K0, ap=.4, alpha=1e-7, k=.6089, rd=500., nu=1e-6):
    """
    Temperature difference at the coral surface relative to the ambient sea-
    water due to the presence of the thermal boundary layer, which increases
    the temperature at the coral tissue.

    Parameters
    ----------
    uc : numeric
        Constricted in-canopy flow [m s^-1].
    Iz : numeric
        Coral surface-averaged light-intensity [mol photons m^-2 s^-1].
    Cf : numeric
        Friction coefficient [-].
    K0 : numeric
        Species and morphology dependent coefficient.
    ap : numeric, optional
        Coral absorptivity [-].
    alpha : numeric, optional
        Thermal diffusivity [m^2 s^-1].
    k : numeric, optional
        Thermal conductivity [J m^-1 s^-1 K^-1].
    rd : numeric, optional
        Velocity boundary layer wall-coordinate.
    nu : numeric, optional
        Kinematic viscosity [m^2 s^-1].

    Returns
    -------
    dT : numeric
        Relative increase of temperature at the coral surface [K].
    """
    deltat = TBL(uc, Cf, rd=rd, nu=nu, alpha=alpha)
    dTc = ((deltat * ap) / (k * K0)) * Iz

    return dTc


# # photosynthesis

def photoacc(I0, Iz, X_old, param, iota=.6, dt=1.,
             Xmax=None, beta=None, output='X_new'):
    """
    Photo-acclimation -- The acclimation of corals to varying light conditions.
    This includes the maximum photosynthetic rate, and the saturation
    intensity. These parameters also depend on their quasi steady-state values,
    which are also light-dependent.

    Parameters
    ----------
    I0 : numeric
        Light-intensity at the surface water (but within the water column)
        [mol photons m-2 s-1].
    Iz : numeric
        Coral biomass-averaged light-intensity [mol photons m^-2 s^-1].
    X_old : numeric
        Saturation light-intensity at previous time-step
        [mol photons m^-2 s^-1].
    param : string
        Parameter of the photo-acclimation of interest.
        Options: ['Ik', 'Pmax'] for saturation intensity and maximum
        photosynthetic rate, respectively.
    iota : numeric, optional
        Acclimation rate [d^-1].
    dt : numeric, optional
        Time-step [d].
    Xmax : numeric, optional
        Maximum value of the quasi steady-state, pre-defined for [param].
    beta : numeric, optional
        Exponent for the quasi steady-state, pre-defined for [param].
    output : string, optional
        Definition of output (see 'Returns'). Options: ['X_new'] and ['X_qss'].

    Returns
    -------
    X_new : numeric
        The value of [param] in the next time-step.
    X_qss : numeric
        The quasi-steady state solution of [param].
    """
    # # input check
    # parameter
    if Xmax is None and beta is None:
        params = ['Ik', 'Pmax']
        if param not in params:
            raise ValueError('Invalid parameter for photo-acclimation. '
                             'Expected one of: {0}'.format(params))
    # output
    outputs = ['X_new', 'X_qss']
    if output not in outputs:
        raise KeyError(
                'Output option not possible. Chosse one of {0}.'
                .format(outputs))

    # # parameter-specific constants
    if Xmax is None:
        if param == 'Ik':
            Xmax = 372.32  # [umol photons s^-1 m^-2]
        elif param == 'Pmax':
            Xmax = 1.43    # [umol O2 s^-1 m^-2]
    if beta is None:
        if param == 'Ik':
            beta = .34     # [-]
        elif param == 'Pmax':
            beta = .09     # [-]

    # # calculations
    # quasi-steady solution
    XS = Xmax * (Iz / I0) ** beta
    if output == 'X_new':
        # differential equation solution
        X_new = XS + (X_old - XS) * np.exp(-iota * dt)

    # # output
    if output == 'X_new':
        out = X_new
    elif output == 'X_qss':
        out = XS

    return out


def thermacc(year, Csp, nn=60, Kvar=2.45, Tunit='deg.C', path=None, tbl=True):
    """
    Thermal-acclimation -- The acclimation of corals to their potentially
    varying thermal environments. This function corrects for the effects of the
    thermal boundary layer if enabled. Otherwise, the thermal-acclimation is
    solely based on the SST time-series.

    Parameters
    ----------
    year : numeric, integer
        Year for which the thermal limits are determined.
    Csp : numeric
        Species constant [-].
    nn : numeric
        Acclimation period [yrs].
    Kvar : numeric, optional
        Variability constant [-].
    Tunit : string, optional
        Unit of temperature used in SST time-series. Options: 'K' and 'deg.C'.
    path : None, string, optional
        Directory to the file with the SST time-series and the light- and flow-
        date (if the use of the thermal boundary layer is enabled).
    tbl : boolean, optional
        Include the effects of the thermal boundary layer.

    Returns
    -------
    Tlo, Thi : array
        Array consisting of the lower and upper limit of the thermal range [K].
    """
    # # input check
    # unit check
    units = ['K', 'deg.C']
    if Tunit not in units:
        raise ValueError(
                'Invalid temperature unit. Expected one of {0}'
                .format(units))
    # availability of SST data
    Tfile = 'TS_SST.txt'
    if path is None:
        Tfilef = Tfile
    else:
        Tfilef = os.path.join(path, Tfile)
    if not os.path.exists(Tfilef):
        raise ValueError(
                'File with SST time-series does not exists or '
                'incorrect directory given: {0}'.format(Tfilef))
    # availability of flow data
    if tbl:
        dTcfile = 'TS_dTc.txt'
        if path is None:
            dTcfilef = dTcfile
        else:
            dTcfilef = os.path.join(path, dTcfile)
        if not os.path.exists(dTcfilef):
            raise ValueError(
                    'File with flow time-series does not exists or '
                    'incorrect directory given: {0}'.format(dTcfilef))

    # # load data
    sst = pd.read_csv(Tfilef, sep='\t')
    sst['date'] = pd.to_datetime(sst['date'])
    if Tunit == 'deg.C':
        sst['sst'] += 273.15
    if tbl:
        dtc = pd.read_csv(dTcfilef, sep='\t')
        dtc['date'] = pd.to_datetime(dtc['date'])

    # # extract historic range applicable to the thermal-acclimation
    Ystart = int(year - Csp * nn)
    sst = sst[np.logical_and(
            sst['date'].dt.year >= Ystart,
            sst['date'].dt.year < year)].reset_index(drop=True)
    if tbl:
        dtc = dtc[np.logical_and(
                dtc['date'].dt.year >= Ystart,
                dtc['date'].dt.year < year)].reset_index(drop=True)

    # # determine representative thermal conditions
    if tbl:
        T = dtc.copy()
        cols = dtc.columns[1:]
        T.set_index('date', inplace=True)
        sst.set_index('date', inplace=True)
        T[cols] += np.tile(sst.values, (1, len(cols)))
    else:
        T = sst.copy()
        T.set_index('date', inplace=True)
        cols = 'sst'

    # # calculations
    # monthly means per year
    MM = T.groupby([T.index.year, T.index.month])[cols].agg(['mean'])
    if tbl:
        MM = MM.droplevel(1, axis=1)
    # annual min. and max. monthly means
    MMM = MM.groupby(level=0).agg(['min', 'max'])
    # statistics of annual min. and max.
    if tbl:
        mmin = MMM.iloc[
                :, MMM.columns.get_level_values(1) == 'min'
                ].droplevel(1, axis=1).mean(axis=0)
        mmax = MMM.iloc[
                :, MMM.columns.get_level_values(1) == 'max'
                ].droplevel(1, axis=1).mean(axis=0)
        smin = MMM.iloc[
                :, MMM.columns.get_level_values(1) == 'min'
                ].droplevel(1, axis=1).std(axis=0)
        smax = MMM.iloc[
                :, MMM.columns.get_level_values(1) == 'max'
                ].droplevel(1, axis=1).std(axis=0)
    else:
        mmin, mmax = MMM.mean(axis=0)
        smin, smax = MMM.std(axis=0)
    # thermal limits
    Tlo = mmin - Kvar * smin
    Thi = mmax + Kvar * smax

    # # output
    return np.array([Tlo, Thi])


def light_eff(Pmax, Iz, I0, Ik):
    """
    Photosynthetic efficiency based on the light conditions. By definition, the
    efficiency has a value between 0 and 1.

    Parameters
    ----------
    Pmax : numeric
        Maximum photosynthetic rate [-].
    Iz : numeric
        Coral biomass-averaged light-intensity [mol photons m^-2 s^-1].
    I0 : numeric
        Light-intensity at the surface water (but within the water column)
        [mol photons m^-2 s^-1].
    Ik : numeric
        Saturation light-intensity [mol photons m^-2 s^-1].

    Returns
    -------
    PI : numeric
        Photo-efficiency [-].
    """
    # # calculations
    try:
        if Ik > 0:
            PI = Pmax * (np.tanh(Iz / Ik) - np.tanh(.01 * I0 / Ik))
        else:
            PI = 0.
    except ValueError:
        PI = np.zeros(len(Ik))
        PI[Ik > 0] = Pmax[Ik > 0] * (np.tanh(Iz[Ik > 0] / Ik[Ik > 0]) -
                                     np.tanh(.01 * I0 / Ik[Ik > 0]))

    # # Output
    return PI


def adapted_temp(Tc, Tc_lo, DT, method='Evenhuis2015'):
    """
    Adapted temperature response; the cubic function that represents the
    thermal response of coral photosynthesis as function of the coral
    temperature.

    Parameters
    ----------
    Tc : numeric
        Coral temperature [K] (forcing).
    Tc_lo : numeric
        Lower bound of coral thermal range [K].
    DT : numeric
        Coral thermal range [K] or [deg.C].
    method : string, optional
        Method used for the specialisation term in the adapted temperature
        response of the coral. Options: ['Evenhuis2015'], ['math']

    Returns
    -------
    f1 : numeric
        Adapted temperature response [-].
    """
    # # input check
    methods = ['Evenhuis2015', 'math']
    if method not in methods:
        raise ValueError('Invalid method. Expected one of: {0}'
                         .format(methods))

    # # functions

    def spec(DT):
        """
        Specialisation term, which rewards the coral with a smaller thermal
        range due to specialisation.
        """
        # # calculations
        if method == 'math':
            sp = 4 * DT ** -4
        elif method == 'Evenhuis2015':
            sp = 4e-4 * np.exp(-.33 * (DT - 10.))

        # # output
        return sp

    # # calculations
    try:
        if Tc > Tc_lo - (1. / np.sqrt(3.)) * DT:
            f1 = - (Tc - Tc_lo) * ((Tc - Tc_lo) ** 2 - DT ** 2)
        else:
            f1 = - ((2 / (3 * np.sqrt(3))) * DT ** 3)
    except ValueError:
        f1 = np.zeros(len(Tc))
        Tcr = Tc_lo - (1. / np.sqrt(3.)) * DT
        f1[Tc > Tcr] = - ((Tc[Tc > Tcr] - Tc_lo[Tc > Tcr]) *
                          ((Tc[Tc > Tcr] - Tc_lo[Tc > Tcr]) ** 2 -
                           DT[Tc > Tcr] ** 2))
        f1[Tc <= Tcr] = - ((2. / (3. * np.sqrt(3))) * DT[Tc <= Tcr] ** 3)
    # incl. specialisation term
    f1 *= spec(DT)

    # # output
    return f1


def therm_env(Tc_lo, DT, Ea=6e4, R=8.31446261815324):
    """
    Thermal envelope response, which accounts for the fact that (bio) chemical
    reactions take place faster at higher temperatures. Follows the Arrhenius
    equation and is calibrated to a temperature of 27 deg.C (or 300 K).

    Parameters
    ----------
    Tc_lo : numeric
        Lower bound of coral thermal range [K].
    DT : numeric
        Cora thermal range [K] or [deg.C].
    Ea : numeric, optional
        Activation energy [J mol^-1]
    R : numeric, optional
        Gas constant [J K^-1 mol^-1] (R = 8.31446261815324 J K^-1 mol^-1)

    Returns
    -------
    f2 : numeric
        Thermal envelope response [-].
    """
    # # calculations
    # optimal temperature
    Topt = Tc_lo + (1. / np.sqrt(3.)) * DT
    # thermal envelope
    f2 = np.exp((Ea / R) * (1. / 300. - 1. / Topt))

    # # output
    return f2


def flow_eff(uc, ucr=.17162374, Pumin=.68886964):
    """
    Photosynthetic efficiency based on flow conditions. By definition, the
    efficiency is a value between 0 and 1.

    Parameters
    ----------
    uc : numeric
        Magnitude of the constricted in-canopy flow velocity [m s^-1].
    ucr : numeric, optional
        Minimum flow velocity at which the photosynthesis is not limited by the
        flow velocity [m s^-1].
    Pumin : numeric, optional
        Minimum flow-based photosynthetic efficiency; the efficiency without
        flow [-].

    Returns
    -------
    Pu : numeric
        Flow-based photosynthetic efficiency [-].
    """
    # # input check
    if Pumin < 0. or Pumin > 1.:
        raise ValueError('Minimum efficiency out of range. '
                         'Give a value between 0 and 1.')

    # # calculations
    Pu = Pumin + (1. - Pumin) * np.tanh(2. * uc / ucr)

    # # output
    return Pu


def photosynthesis(I0, Iz, Ik, Pmax,
                   Tc, Tc_lo, DT,
                   U,
                   Ea=6e4, R=8.31446261815324, method='Evenhuis2015',
                   ucr=.17162374, Pumin=.68886964):
    """
    Photosynthetic response as function of the light-intensity and the coral
    temperature. The described photosynthetic response is a proxy of the
    photosynthetic rate of the coral-symbiont.

    Parameters
    ----------
        LIGHT
    I0 : numeric
        Light-intensity at the surface water (but within the water column)
        [mol photons m-2 s-1].
    Iz : numeric
        Coral biomass-averaged light-intensity [mol photons m^-2 s^-1].
    Pmax : numeric
        Maximum photosynthetic rate [mol O2 s^-1 m^-2].
    Ik : numeric
        Saturation light-intensity [mol photons m^-2 s^-1].

        TEMPERATURE
    Tc : numeric
        Coral temperature [K].
    Tc_lo : numeric
        Lower bound coral thermal range [K].
    DT : numeric
        Coral thermal range [K] or [deg.C].
    Ea : numeric, optional
        Activation energy [J mol^-1]
    R : numeric, optional
        Gas constant [J K^-1 mol^-1] (R = 8.31446261815324 J K^-1 mol^-1)
    method : string, optional
        Method used for the specialisation term in the adapted temperature
        response of the coral. Options: ['Evenhuis2015'], ['math']

        FLOW
    U : numeric
        Flow velocity [m s^-1].
    ucr : numeric, optional
        Minimum flow velocity at which the photosynthesis is not limited by the
        flow velocity [m s^-1].
    Pumin : numeric, optional
        Minimum flow-based photosynthetic efficiency; the efficiency without
        flow [-].

    Returns
    -------
    P : numeric
        proxy of photosynthetic rate [-].
    """
    # # input check
    methods = ['Evenhuis2015', 'math']
    if method not in methods:
        raise ValueError('Invalid method. Expected one of: {0}'
                         .format(methods))

    # # calculations
    # light, P(I)
    PI = light_eff(Pmax, Iz, I0, Ik)
    # temperature
    # -> adapted temperature response, f1
    f1 = adapted_temp(Tc, Tc_lo, DT, method=method)
    # -> thermal envelope, f2
    f2 = therm_env(Tc_lo, DT, Ea=Ea, R=R)
    # -> combined, P(T)
    PT = f1 * f2
    # flow
    PU = flow_eff(U, ucr, Pumin)
    # combined, P(I,T, u)
    P = PI * PT * PU

    # # output
    return P


# # population states


def pop_states(PS, K, Csp, P0,
               rG=.002, rR=.2, rM=.04, rB=8., dt=1., tsh=0.):
    """
    Population dynamics expressed in population states: (1) healthy population;
    (2) recovering population; (3) pale population; and (4) bleached
    population. The check whether the conditions are enhancing growth or
    bleaching is incorporated in the photosynthetic rate, which is positive
    when the conditions enhance growth; and negative during bleaching
    conditions.

    Parameters
    ----------
    PS : time-series
        Proxy of the photosynthetic rate as function of time.
    K : numeric
        Carrying capacity of the substratum [m^2 m^-2].
    Csp : numeric
        Species constant.
    P0 : array
        Array consisting of the initial condition of the distribution of the
        population states, P = [P_H, P_R, P_P, P_B]. (P_H: healthy pop.; P_R:
        recovering pop.; P_P: pale pop.; and P_B: bleached pop.)
    rG : numeric, optional
        Growth rate [d^-1].
    rR : numeric, optional
        Recovering rate [d^-1].
    rM : numeric, optional
        Mortality rate [d^-1].
    rB : numeric, optional
        Bleaching rate [d^-1].
    dt : numeric, optional
        Time-step [d].
    tsh : numeric, optional
        Threshold below which corals start to bleach, expressed in the proxy
        for photosynthetic rate, PS.

    Returns
    -------
    P : time-series
        Time-series of the distribution of the population states [m^2 m^-2].
        Shape of martix: [len(<time-series>) x 4].
    """
    # # calculations
    P = np.zeros((len(PS), 4))
    # growing conditions
    # > bleached pop.
    P[PS > tsh, 3] = (
            P0[PS > tsh, 3] / (1. + dt * (
                    (8. * rR * PS[PS > tsh]) / Csp + rM * Csp)))
    # > pale pop.
    P[PS > tsh, 2] = (
            (P0[PS > tsh, 2] + (
                    8. * dt * rR * PS[PS > tsh] / Csp) *
             P[PS > tsh, 3]) / (1. + dt * rR * PS[PS > tsh] * Csp))
    # > recovering pop.
    P[PS > tsh, 1] = (
            (P0[PS > tsh, 1] +
             dt * rR * PS[PS > tsh] * Csp * P[PS > tsh, 2]) /
            (1. + .5 * dt * rR * PS[PS > tsh] * Csp))
    # > healthy pop.
    a = dt * rG * PS[PS > tsh] * Csp / K[PS > tsh]
    b = (1. - dt * rG * PS[PS > tsh] * Csp * (
            1. - P[PS > tsh, 1:].sum(axis=1) / K[PS > tsh]))
    c = - (P0[PS > tsh, 0] +
           .5 * dt * rR * PS[PS > tsh] * Csp * P[PS > tsh, 1])
    P[PS > tsh, 0] = (-b + np.sqrt(b ** 2 - 4. * a * c)) / (2. * a)

    # bleaching conditions
    # > healthy pop.
    P[PS <= tsh, 0] = (
            P0[PS <= tsh, 0] /
            (1. - dt * rB * PS[PS <= tsh] * Csp))
    # > recovering pop.
    P[PS <= tsh, 1] = (
            P0[PS <= tsh, 1] /
            (1. - dt * rB * PS[PS <= tsh] * Csp))
    # > pale pop.
    P[PS <= tsh, 2] = (
            (P0[PS <= tsh, 2] - dt * rB * PS[PS <= tsh] * Csp * (
                    P[PS <= tsh, 0] + P[PS <= tsh, 1])) /
            (1. - .5 * dt * rB * PS[PS <= tsh] * Csp))
    # > bleached pop.
    P[PS <= tsh, 3] = (
            (P0[PS <= tsh, 3] -
             .5 * dt * rB * PS[PS <= tsh] * Csp * P[PS <= tsh, 2]) /
            (1. - .25 * dt * rB * PS[PS <= tsh] * Csp))

    # # check on carrying capacity
    if any(P.sum(axis=1) > 1.0001 * K):
        print('WARNING: PT > {0} at location(s) {1} (PT = {2})'
              .format(K[P.sum(axis=1) > 1.0001 * K],
                      np.arange(len(K))[P.sum(axis=1) > 1.0001 * K],
                      P[P.sum(axis=1) > 1.0001 * K]))

    # # output
    return P


# # calcification


def calcification(PS, omega_a, PH, Csp,
                  omega_0=None, ka=None, gC=.5, methodARG='MM'):
    """
    Calcification rate -- The combining effects of the photosynthesis, the
    aragonite saturation state, etc. on the calcification rate, which
    eventually determines the growth rate of the coral.

    Parameters
    ----------
    PS : numeric or array
        Proxy of photosynthetic rate [-].
    omega_a : numeric or array
        Aragonite saturation state [-].
    PH : numeric or array
        Healthy population [m^2 m^-2].
    Csp : numeric
        Species constant [-].
    omega_0 : numeric, optional
        Aragonite saturation state at which the function crosses zero [-].
    ka : numeric, optional
        Michaelis constant, which is defined such that the rate is half the
        maximum rate at omega_a = omega_0 + ka [-].
        In case of the tangent-hyperbolic function, it describes a comparable
        definition of the slope of the function.
    gC : numeric, optional
        Calcification constant [kg m^-2 d^-1].
    methodARG: string, optional
        Method used in aragonite dependency determination.
        Options: ['MM', 'tanh'], representing the Michaelis-Menten equation and
        the tangent-hyperbolic function.

    Returns
    -------
    G : numeric or array
        Calcification rate [kg m^-2 d^-1].
    """
    # # functions

    def gamma_arag(omega_a, omega_0, ka):
        """
        Representation of the effect of the aragonite saturation state on the
        calcification rate.
        """
        # # constants based on method
        if omega_0 is None and ka is None:
            if methodARG == 'MM':
                omega_0 = .14587415
                ka = .66236107
            elif methodARG == 'tanh':
                omega_0 = 1.65357640e-16
                ka = 2.14147665

        # # calculations
        if methodARG == 'MM':
            gamma = (omega_a - omega_0) / (ka + omega_a - omega_0)
        elif methodARG == 'tanh':
            gamma = np.tanh((omega_a - omega_0) / ka)

        # # output
        return gamma

    # # calculations
    gamma = gamma_arag(omega_a, omega_0=omega_0, ka=ka)
    G = gC * Csp * PH * gamma * PS
    try:
        if G < 0:
            G = 0
    except ValueError:
        G[G < 0] = 0
    return G


# # coral growth

def coral_volume(dc, hc, bc, tc):
    """
    Translation between morphological dimensions and the coral volume.

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
    Vc : numeric
        Coral volume [m^3].
    """
    # # calculations
    Vc = .25 * np.pi * ((hc - tc) * (bc ** 2) + tc * (dc ** 2))

    # # output
    return Vc


def morph_update(dc, hc, bc, tc, ac, Gsum, h, Kd, Iz, I0, uc,
                 Xf=.1, Xp=.5, Xpu=.1, Xs=.5/np.sqrt(2.), XsI=.1, Xsu=.1,
                 ucr=.172, thetamax=.5*np.pi, rhoc=1600., dtyear=1.,
                 recruitment=False):
    """
    Update of the morphological dimensions based on the old morphological
    ratios and coral volume and the optimal morphological ratios and the added
    coral volume due to calcification.

    The parameters (see below) are subdivided in five groups: (1) morphological
    dimensions; (2) rate of change, pt. I; (3) proportionality constants;
    (4) rate of change, pt. II; and (5) recruitment.

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
    ----------
    Gsum : numeric
        Annual calcification rate [kg m^-2 y^-1].
        If recruitment is enabled, it represents the volumetric addition due to
        the recruitment [m^3].
    h : numeric
        Water depth [m].
    Kd : numeric
        Light-attenuation coefficient [m^-1].
    Iz : numeric
        Representative light-intensity at depth z [mol photons m^-2 s^-1].
    I0 : numeric
        Incoming light-intensity [mol photons m^-2 s^-1].
    uc : numeric
        Constricted in-canopy flow [m s^-1].
    ----------
    Xf : numeric, optional
        Overall form proportionality constant [-]. (To be fitted.)
    Xp : numeric
        Overall plate proportionality constnat [-].
    Xpu : numeric
        Flow plate porportionality constant [-]. (To be fitted.)
    Xs : numeric, optional
        Overall spacing proportionality constant [-].
    XsI : numeric, optional
        Light spacing proportionality constant [-]. (To be fitted.)
    Xsu : numeric, optional
        Flow spacing proportionality constant [-]. (To be fitted.)
    ucr : numeric, optional
        Critical flow velocity (to be fitted) [m s^-1].
    ----------
    thetamax : numeric, optional
        Maximum spreading of light [rad].
    rhoc : numeric, optional
        Coral density [kg m^-3].
    dtyear : numeric, optional
        Time-step [y].
    recruitment : boolean, optional
        Morphological update after recruitment.

    Returns
    -------
    dc1 : numeric
        Diameter of the plate of the coral [m]. (New.)
    hc1 : numeric
        Coral height [m]. (New.)
    bc1 : numeric
        Diameter of the base of the coral [m]. (New.)
    tc1 : numeric
        Thickness of the plate of the coral [m]. (New.)
    ac1 : numeric
        Axial distance between corals [m]. (New.)
    """
    # # fucntions

    def rf_opt(Iz, I0, uc):
        """
        The optimal form ratio (height-to-diameter) based on the light- and
        flow-conditions that the coral colony encounters.
        """
        # # calculations
        rf = Xf * (Iz / I0) * (ucr / uc)

        # # output
        return rf

    def rp_opt(uc):
        """
        The optimal plate ratio (base-to-top diameter) based on the flow-
        conditions that the coral encounters.
        """
        # # calculations
        rp = Xp * (1. + np.tanh(Xpu * (uc - ucr) / ucr))

        # # output
        return rp

    def rs_opt(Iz, I0, uc):
        """
        The optimal spacing ratio (diameter-to-axial distance) based on the
        light- and flow-conditions that the coral encounters.
        """
        # # calculations
        rsI = (1. - np.tanh(XsI * Iz / I0))
        rsu = (1. + np.tanh(Xsu * (uc - ucr) / ucr))
        rs = Xs * rsI * rsu

        # # output
        return rs

    def delta_Vc(dc, hc, bc, tc, ac, Gsum, h, Kd):
        """
        Addition to coral volume due to calcification.
        """
        # # calculations
        # biomass
        Bc = biomass(dc, hc, bc, tc, h, Kd, thetamax=thetamax)
        # volume increase
        dVc = .5 * ac ** 2 * Gsum * dtyear / rhoc * Bc

        # # output
        return dVc

    def ratio_update(r_old, ropt, Vc, dVc):
        """
        Update the morphological ratio (rf, rp or rs) by solving the
        differential equation, which is based on a weighted-average of the
        morphological ratio by its volume.
        """
        # # calculations
        r = (Vc * r_old + dVc * ropt) / (Vc + dVc)

        # # output
        return r

    def Vc2dc(Vc, rf, rp):
        """
        Translation from coral volume and morphological ratios to the diameter
        of the plate of the coral.
        """
        # # calculations
        dc = ((4. * Vc) /
              (np.pi * rf * rp * (1. + rp - rp ** 2))) ** (1. / 3.)

        # # output
        return dc

    def Vc2hc(Vc, rf, rp):
        """
        Translation from coral volume and morphological ratios to the coral
        height.
        """
        # # calculations
        hc = ((4. * Vc * rf ** 2) /
              (np.pi * (rp + rp ** 2 - rp ** 3))) ** (1. / 3.)

        # # output
        return hc

    def Vc2bc(Vc, rf, rp):
        """
        Translation from coral volume and morphological ratios to the diameter
        of the base of the coral.
        """
        # # calculations
        bc = ((4. * Vc * rp ** 2) /
              (np.pi * rf * (1. + rp - rp ** 2))) ** (1. / 3.)

        # # output
        return bc

    def Vc2tc(Vc, rf, rp):
        """
        Translation from coral volume and morphological ratios to the thickness
        of the plate.
        """
        # # calculations
        tc = ((4. * Vc * rf ** 2 * rp ** 2) /
              (np.pi * (1 + rp - rp ** 2))) ** (1. / 3.)

        # # output
        return tc

    def Vc2ac(Vc, rf, rp, rs):
        """
        Translation from coral volume and morphological ratios to the axial
        distance between corals.
        """
        # # calculations
        ac = (1. / rs) * ((4. * Vc) /
                          (np.pi * rf * (rp + rp ** 2 - rp ** 3))) ** (1. / 3.)

        # # output
        return ac

    # # calculations
    # coral volume
    Vc = coral_volume(dc, hc, bc, tc)
    if recruitment:
        dVc = Gsum
    else:
        dVc = delta_Vc(dc, hc, bc, tc, ac, Gsum, h, Kd)
    Vc1 = Vc + dVc
    # old morphological ratios
    try:
        if dc > 0:
            rf = hc / dc
            rp = bc / dc
        else:
            rf = 0.
            rp = 0.
    except ValueError:
        rf = np.zeros(len(dc))
        rf[dc > 0] = hc[dc > 0] / dc[dc > 0]
        rp = np.zeros(len(dc))
        rp[dc > 0] = bc[dc > 0] / dc[dc > 0]
    try:
        if ac > 0:
            rs = dc / ac
        else:
            rs = 0.
    except ValueError:
        rs = np.zeros(len(ac))
        rs[ac > 0] = dc[ac > 0] / ac[ac > 0]
    # optimal morphological ratios
    rfopt = rf_opt(Iz, I0, uc)
    rpopt = rp_opt(uc)
    rsopt = rs_opt(Iz, I0, uc)
    # new morphological ratios
    rf1 = ratio_update(rf, rfopt, Vc, dVc)
    rp1 = ratio_update(rp, rpopt, Vc, dVc)
    rs1 = ratio_update(rs, rsopt, Vc, dVc)
    # new morphological dimensions
    dc1 = Vc2dc(Vc1, rf1, rp1)
    hc1 = Vc2hc(Vc1, rf1, rp1)
    bc1 = Vc2bc(Vc1, rf1, rp1)
    tc1 = Vc2tc(Vc1, rf1, rp1)
    ac1 = Vc2ac(Vc1, rf1, rp1, rs1)

    # # output
    return np.array([Vc1, dc1, hc1, bc1, tc1, ac1])


def dc_rep(dc, hc, bc, tc):
    """
    Translation from coral morphology parameters to representative coral
    diameter as used in the hydrodynamic model.

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

    Returns
    -------
    dc_av : numeric
        Representative coral diameter [m].
    """
    # # calculations
    try:
        if dc > 0:
            dc_av = (bc * (hc - tc) + dc * tc) / hc
        else:
            dc_av = 0.
    except ValueError:
        dc_av = np.zeros(len(dc))
        dc_av[dc > 0] = (bc[dc > 0] * (hc[dc > 0] - tc[dc > 0]) +
                         dc[dc > 0] * tc[dc > 0]) / hc[dc > 0]

    # # output
    return dc_av


def morph2vegden(dc, hc, bc, tc, ac):
    """
    Translation from coral morphology parameters to parameters as used for the
    modelling of vegetation; i.e. rewrite the coral morphology into the
    vegetation density.

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
    rnveg : numeric
        Vegetation density [stems m^-2].
    """
    # # calculations
    # average diameter
    dc_av = dc_rep(dc, hc, bc, tc)
    # representative vegetation density
    try:
        if ac > 0:
            rnveg = (2 * dc_av) / (ac ** 2)
        else:
            rnveg = 0.
    except ValueError:
        rnveg = np.zeros(len(ac))
        rnveg[ac > 0] = (2. * dc_av[ac > 0]) / (ac[ac > 0] ** 2)

    # # output
    return rnveg


# # storm damage


def DMT(ub, sigmat, Cd=1., rhow=1025):
    """
    The dislodgement mechanical threshold (DMT) as part of the threshold of
    dislodgement. When the DMT is smaller than the colony shape factor (CSF),
    the coral colony is dislodged.

    Parameters
    ----------
    ub : numeric, array
        Maximum bulk flow velocity [m s^-1].
    sigmat : numeric, array
        Tensile strength at the coral-substrate interface [N m^-2].
    Cd : numeric, array, optional
        Drag coefficient [-].
    rhow : numeric, array, optional
        Density of water [kg m^-3].

    Returns
    -------
    DMT : numeric, array
        The dislodgement mechanical threshold [-].
    """
    # # calculations
    try:
        if ub > 0:
            DMT = sigmat / (rhow * Cd * ub ** 2)
        else:
            DMT = 1e20
    except ValueError:
        DMT = np.zeros(len(ub))
        DMT[ub > 0] = sigmat / (rhow * Cd * ub[ub > 0] ** 2)

    # # output
    return DMT


def CSF(dc, hc, bc, tc):
    """
    The colony shape factor (CSF) as part of the thershold of dislodgement.
    When the CSF is larger than the dislodgement mechanical threshold (DMT),
    the coral colony is dislodged.

    Parameters
    ----------
    hc : numeric, array
        Height of full coral colony [m].
    dc : numeric, array
        Diameter of plate [m].
    bc : numeric, array
        Diameter of base/foot [m].
    tc : numeric, array
        Thickness plate [m].

    Returns
    -------
    CSF : numeric, array
        The colony shape factor [-].
    """
    # # calculations
    # arms of moment
    at = hc - .5 * tc
    ab = .5 * (hc - tc)
    # area of moment
    At = dc * tc
    Ab = bc * (hc - tc)
    # integral
    S = at * At + ab * Ab
    # colony shape factor
    try:
        if bc > 0:
            CSF = 16. / (np.pi * bc ** 3) * S
        else:
            CSF = 0.
    except ValueError:
        CSF = np.zeros(bc.shape)
        CSF[bc > 0] = 16. / (np.pi * bc[bc > 0] ** 3) * S[bc > 0]

    # # output
    return CSF


def spawning(Nl, ps, dl, PHav, PT, K, param):
    """
    The contribution to the coral cover or coral volume due to mass spawning.

    Parameters
    ----------
    Nl : numeric
        Number of larvae [-].
    ps : numeric
        Probability of settlement [-].
    dl : numeric
        Larval diameter [-].
    PHav : numeric
        Reef-averaged healthy coral cover [-].
    PT : numeric
        Total population [-].
    K : numeric
        Carrying capacity [-].
    param : string
        Definition of contribution to (1) coral cover; or (2) coral volume.

    Returns
    -------
    L : numeric
        Addition due to spawning
    """
    # # input check
    params = ['P', 'V']
    if param not in params:
        raise KeyError(
                'Definition of parameter not in options: {0}'.format(params))

    # # calculations
    # potential of coral recruitment
    if param == 'P':
        S = ps * Nl * dl ** 2
    elif param == 'V':
        S = ps * Nl * dl ** 3
    # coral recruitment
    L = S * PHav * (1. - PT / K)

    # # output
    return L


# # model check
print('Functions defined.\n')


# =============================================================================
# # # # initialisation of vegetation variables
# =============================================================================
# # morphological dimensions
# diameter plate
dc = K * dc0
# coral height ~ stemheight
hc = K * hc0
stemheight = modelFM.get_var('stemheight')
stemheight[range(ndxi)] = K * hc0
modelFM.set_var('stemheight', stemheight)
# diameter base
bc = K * bc0
# thickness plate
tc = K * tc0
# axial distance
ac = K * ac0
# coral volume
Vc = coral_volume(dc, hc, bc, tc)
# representative diameter
diaveg = modelFM.get_var('diaveg')
diaveg[range(ndxi)] = K * dc_rep(dc0, hc0, bc0, tc0)
modelFM.set_var('diaveg', diaveg)
# 'vegetation' density
rnveg = modelFM.get_var('rnveg')
rnveg[range(ndxi)] = K * morph2vegden(dc0, hc0, bc0, tc0, ac0)
modelFM.set_var('rnveg', rnveg)


# =============================================================================
# # # # run the model
# =============================================================================
print('Start time : {0}\n'.format(datetime.datetime.now().time()))

with tqdm(range(int(Y))) as pbar:
    for i in pbar:
        # # update hydrodynamic model
        pbar.set_postfix(inner_loop='update Delft3D')
        modelDIMR.update(mtpervt)

        # # extract variables from DFM via BMI
        pbar.set_postfix(inner_loop='extract variables')
        # flow characteristics
        is_sumvalsnd = modelFM.get_var('is_sumvalsnd')
        is_maxvalsnd = modelFM.get_var('is_maxvalsnd')
        is_dtint = modelFM.get_var('is_dtint')
        uwav = modelFM.get_var('Uorb')[range(ndxi)]
        twav = modelFM.get_var('twav')[range(ndxi)]
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
        vel_mean = is_sumvalsnd[range(ndxi), 1] / is_dtint
        wd_mean = is_sumvalsnd[range(ndxi), 2] / is_dtint

        # # working arrays
        pbar.set_postfix(inner_loop='set working arrays')
        # length of the year (number of days)
        No_days = len(T[T.index.year == years[i]])
        # representative light-intensity
        Iz = np.zeros((len(K), int(No_days)))
        # saturation intensity
        Ik = np.zeros((len(K), int(No_days)))
        # max. photosynthesis
        Pmax = np.zeros((len(K), int(No_days)))
        # coral temperature
        Tc = np.zeros((len(K), int(No_days)))
        # photosynthesis
        PS = np.zeros((len(K), int(No_days)))
        # population states
        P = np.zeros((len(K), int(No_days), 4))
        # calcification
        G = np.zeros((len(K), int(No_days)))
        # morphological dimensions
        M = np.zeros((6, len(K)))
        # spawning - cora volume
        Svol = np.zeros(len(K))
        # incoming light of years[i]
        I0y = I0['par'][I0.index.year == years[i]].values
        # light-attenuation coefficient of years[i]
        if os.path.exists(os.path.join(inputdir, 'TS_Kd.txt')):
            Kdy = Kd['kd'][Kd.index.year == years[i]].values
        else:
            Kdy = Kd0 * np.ones(int(No_days))
        # SST of years[i]
        Ty = T['sst'][T.index.year == years[i]].values
        # aragonite saturation state
        if os.path.exists(os.path.join(inputdir, 'TS_ARG.txt')):
            omegay = omega['arg'][omega.index.year == years[i]].values
        else:
            omegay = omega0 * np.ones(int(No_days))

        # # representative environment
        # representative light-intensity { thetamax = 90. [deg] }
        pbar.set_postfix(inner_loop='coral environment - light')
        for j in range(int(No_days)):
            Iz[K > 0, j] = av_light(
                    I0y[j], Kdy[j], wd_mean[K > 0],
                    dc[K > 0], hc[K > 0], bc[K > 0], tc[K > 0])
        # in-canopy flow
        pbar.set_postfix(inner_loop='coral environment - flow')
        if cft:
            alphaw = np.ones(ndxi)
            alphac = np.ones(ndxi)
            for x in np.arange(ndxi)[K > 0]:
                alphaw[x] = drag(
                        wd_mean[x], uwav[x], twav[x],
                        dcmodel[x], hc[x], ac[x],
                        Cs, Cm, output='aw')
                alphac[x] = drag(
                        wd_mean[x], vel_mean[x], 1e3,
                        dcmodel[x], hc[x], ac[x],
                        Cs, Cm, output='aw')
            ucm = wavecurrent(uwav, vel_mean, wcangle,
                              alphaw=alphaw, alphac=alphac)
            if tbl:
                # write annual representative in-canopy flow to file
                dTcfile = os.path.join(inputdir, 'TS_dTc.txt')
                # > write historic file
                if i == 0:
                    pbar.set_postfix(
                            inner_loop=('coral environment - '
                                        'thermal-acclimation - '
                                        'write historic file'))
                    dtdates = T.index[T.index.year < years[i]]
                    I0tbl = I0[I0.index.year < years[i]].values
                    dTc = np.zeros((len(dtdates), ndxi))
                    for j in range(len(dtdates)):
                        Iztbl = I0tbl[j] * np.exp(-Kdy[0] * wd_mean)
                        dTc[j, :] = delta_Tc(
                                ucm, Iztbl, Cf, K0)
                    df = pd.DataFrame(dTc, columns=np.arange(ndxi))
                    df.set_index(dtdates, inplace=True)
                    df.to_csv(dTcfile, sep='\t', mode='w')
                    # > > remove working arrays
                    del dtdates, I0tbl, Iztbl, dTc, df
                # > append to historic file
                dTc = np.zeros((int(No_days), ndxi))
                for j in range(int(No_days)):
                    dTc[j, :] = delta_Tc(
                            ucm, Iz[:, j], Cf, K0)
                df = pd.DataFrame(dTc, columns=np.arange(ndxi))
                df.set_index(T.index[T.index.year == years[i]], inplace=True)
                df.to_csv(dTcfile, header=None, sep='\t', mode='a')
                # > > remove working arrays
                del dTc, df
            # fitting parameters
            Pumin = .6889
            ucr = .1716
        else:
            ucm = wavecurrent(uwav, vel_mean, wcangle)
            # fitting parameters
            Pumin = .6889
            ucr = .5173
        # representative (coral) temperature [K]
        pbar.set_postfix(inner_loop='coral environment - temperature')
        # { ap=.4, alpha=1e-7, k=.6089, rd=500., nu=1e-6 }
        if tbl:
            for j in range(int(No_days)):
                Tc[K > 0, j] = Ty[j] + delta_Tc(
                        ucm[K > 0], Iz[K > 0, j], Cf, K0) + 273.15
        else:
            Tc = Ty + 273.15

        # # coral physiology
        # photo-acclimation
        pbar.set_postfix(inner_loop='coral physiology - photo-acclimation')
        for j in range(int(No_days)):
            Ik[K > 0, j] = photoacc(
                    I0y[j], Iz[K > 0, j], None,
                    'Ik', Xmax=Ikmax, output='X_qss')
            Pmax[K > 0, j] = photoacc(
                    I0y[j], Iz[K > 0, j], None,
                    'Pmax', Xmax=1., output='X_qss')
        # thermal-acclimation
        pbar.set_postfix(inner_loop='coral physiology - thermal-acclimation')
        Tlo, Thi = thermacc(years[i], Csp, Kvar=Kvar, path=inputdir, tbl=tbl)
        DT = Thi - Tlo
        # photosynthesis
        pbar.set_postfix(inner_loop='coral physiology - photosynthesis')
        if tbl:
            for j in range(int(No_days)):
                PS[K > 0, j] = photosynthesis(
                        I0y[j], Iz[K > 0, j], Ik[K > 0, j], Pmax[K > 0, j],
                        Tc[K > 0, j], Tlo[K > 0], DT[K > 0],
                        ucm[K > 0], Pumin=Pumin, ucr=ucr)
        else:
            for j in range(int(No_days)):
                PS[K > 0, j] = photosynthesis(
                        I0y[j], Iz[K > 0, j], Ik[K > 0, j], Pmax[K > 0, j],
                        Tc[j], Tlo, DT, ucm[K > 0], Pumin=Pumin, ucr=ucr)
        # population dynamics
        pbar.set_postfix(inner_loop='coral physiology - population dynamics')
        for j in range(int(No_days)):
            if j == 0:
                P[K > 0, j, :] = pop_states(
                        PS[K > 0, j], K[K > 0], Csp, P0[K > 0, :])
            else:
                P[K > 0, j, :] = pop_states(
                        PS[K > 0, j], K[K > 0], Csp, P[K > 0, j - 1, :])
        P0[K > 0, :] = P[K > 0, -1, :]
        # calcification
        pbar.set_postfix(inner_loop='coral physiology - calcification')
        for j in range(int(No_days)):
            G[K > 0, j] = calcification(
                    PS[K > 0, j], omegay[j], P[K > 0, j, 0], Csp)

        # # coral morphology
        pbar.set_postfix(inner_loop='coral morphology')
        # update morphology
        M[:, K > 0] = morph_update(
                dc[K > 0], hc[K > 0], bc[K > 0], tc[K > 0], ac[K > 0],
                G.sum(axis=1)[K > 0], wd_mean[K > 0], Kdy.mean(),
                Iz.mean(axis=1)[K > 0], I0y.mean(), ucm[K > 0],
                Xf=Xf, Xpu=Xpu, XsI=XsI, Xsu=Xsu)
        Vc, dc, hc, bc, tc, ac = M
        # translate to model parameters - matrix shape & value
        dcmodel[K > 0] = dc_rep(
                dc[K > 0], hc[K > 0], bc[K > 0], tc[K > 0])
        diaveg[range(ndxi)] = dcmodel
        stemheight[range(ndxi)] = hc
        rnveg[range(ndxi)] = morph2vegden(
                dc, hc, bc, tc, ac)

        # # show intermediate model output

        # # return coral data to hydrodynamic model
        pbar.set_postfix(inner_loop='reset counters and return to Delft3D')
        # reset counters
        is_sumvalsnd.fill(0.)
        is_maxvalsnd.fill(0.)
        is_dtint.fill(0.)
        # push counters and updated coral field to model
        modelFM.set_var('is_sumvalsnd', is_sumvalsnd)
        modelFM.set_var('is_maxvalsnd', is_maxvalsnd)
        modelFM.set_var('is_dtint', is_dtint)
        modelFM.set_var('rnveg', rnveg)
        modelFM.set_var('diaveg', diaveg)
        modelFM.set_var('stemheight', stemheight)

        # # storm damage
        # storm check
        if H['stormcat'][H.index == years[i]].values > 0:
            # run storm conditions
            pbar.set_postfix(inner_loop='STORM - update Delft3D')
            modelDIMR.update(mtpervt_storm)
            # extract variables from DFM via BMI
            pbar.set_postfix(inner_loop='STORM - extract variables')
            is_sumvalsnd = modelFM.get_var('is_sumvalsnd')
            is_maxvalsnd = modelFM.get_var('is_maxvalsnd')
            is_dtint = modelFM.get_var('is_dtint')
            uwav = modelFM.get_var('Uorb')[range(ndxi)]
            rnveg = modelFM.get_var('rnveg')
            diaveg = modelFM.get_var('diaveg')
            stemheight = modelFM.get_var('stemheight')
            rnveg = modelFM.get_var('rnveg')
            # maximum flow velocity
            vel_max = is_maxvalsnd[range(ndxi), 1]
            ummax = wavecurrent(uwav, vel_max, wcangle)
            # Dislodgement Mechanical Threshold (DMT)
            pbar.set_postfix(inner_loop='STORM - dislodgement criterion')
            dmt = DMT(ummax, sigmat)
            # Colony Shape Factor (CSF)
            csf = CSF(dc, hc, bc, tc)
            # dislodgement criterion
            Vc[dmt < csf] = 0.
            P0[dmt < csf, :] = 0.
            # update morphological dimensions
            pbar.set_postfix(inner_loop='STORM - update morphology')
            dc[dmt < csf] = 0.
            hc[dmt < csf] = 0.
            bc[dmt < csf] = 0.
            tc[dmt < csf] = 0.
            ac[dmt < csf] = 0.
            # update model parameters
            dcmodel[K > 0] = dc_rep(
                    dc[K > 0], hc[K > 0], bc[K > 0], tc[K > 0])
            diaveg[range(ndxi)] = dcmodel
            stemheight[range(ndxi)] = hc
            rnveg[range(ndxi)] = morph2vegden(
                    dc, hc, bc, tc, ac)
            # reset counters
            pbar.set_postfix(
                    inner_loop='STORM - reset counters and return to Delft3D')
            is_sumvalsnd.fill(0.)
            is_maxvalsnd.fill(0.)
            is_dtint.fill(0.)
            # push counters and updated coral field to model
            modelFM.set_var('is_sumvalsnd', is_sumvalsnd)
            modelFM.set_var('is_maxvalsnd', is_maxvalsnd)
            modelFM.set_var('is_dtint', is_dtint)
            modelFM.set_var('diaveg', diaveg)
            modelFM.set_var('stemheight', hc)
            modelFM.set_var('rnveg', rnveg)

        # # recruitment / spawning
        pbar.set_postfix(inner_loop='coral recruitment')
        # coral volume
        Svol[K > 0] = spawning(
                Nl, ps, dl, P[K > 0, -1, 0].mean(),
                P[K > 0, -1, :].sum(axis=1), K[K > 0], 'V')
        # update morphology
        M[:, K > 0] = morph_update(
                dc[K > 0], hc[K > 0], bc[K > 0], tc[K > 0], ac[K > 0],
                Svol[K > 0], wd_mean[K > 0], Kdy.mean(),
                Iz.mean(axis=1)[K > 0], I0y.mean(), ucm[K > 0],
                Xf=Xf, Xpu=Xpu, XsI=XsI, Xsu=Xsu,
                recruitment=True)
        Vc, dc, hc, bc, tc, ac = M
        # translate to model parameters - matrix shape & value
        dcmodel[K > 0] = dc_rep(
                dc[K > 0], hc[K > 0], bc[K > 0], tc[K > 0])
        diaveg[range(ndxi)] = dcmodel
        stemheight[range(ndxi)] = hc
        rnveg[range(ndxi)] = morph2vegden(
                dc, hc, bc, tc, ac)
        # coral cover
        P0[K > 0, 0] += spawning(
                Nl, ps, dl, P[K > 0, -1, 0].mean(),
                P[K > 0, -1, :].sum(axis=1), K[K > 0], 'P')

# =============================================================================
#       # # # export model results
# =============================================================================
        # # map-file
        pbar.set_postfix(inner_loop='write file - map')
        files = [U2mfile, T2mfile, PS2mfile, P2mfile, G2mfile, M2mfile]
        if any(files):
            if i == 0:
                mset = Dataset(mapfilef, 'w', format='NETCDF4')
                mset.description = 'Mapped simulation data of the coral_model'

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

                if U2mfile:
                    Uset = mset.createVariable('ucm', 'f8',
                                               ('time', 'nmesh2d_face'))
                    Uset.long_name = 'in-canopy flow'
                    Uset.units = 'm s^-1'
                if T2mfile:
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
                if PS2mfile:
                    PSset = mset.createVariable('PS', 'f8',
                                                ('time', 'nmesh2d_face'))
                    PSset.long_name = 'annual mean photosynthesis'
                    PSset.units = '-'
                if P2mfile:
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
                if G2mfile:
                    Gset = mset.createVariable('G', 'f8',
                                               ('time', 'nmesh2d_face'))
                    Gset.long_name = 'calcification'
                    Gset.units = 'kg m^-2 y^-1'
                if M2mfile:
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

                # data
                t[:] = np.array([years[i] - 1, years[i]])
                x[:] = xzw[range(ndxi)]
                y[:] = yzw[range(ndxi)]
                if U2mfile:
                    Uset[:, :] = np.array([np.zeros(ndxi), ucm])
                if T2mfile:
                    if tbl:
                        Tset[:, :] = np.array([np.zeros(ndxi), Tc[:, -1]])
                        Tloset[:, :] = np.array([np.zeros(ndxi), Tlo])
                        Thiset[:, :] = np.array([np.zeros(ndxi), Thi])
                    else:
                        Tset[:, :] = np.array([np.zeros(ndxi),
                                               Tc[-1] * np.ones(ndxi)])
                        Tloset[:, :] = np.array([np.zeros(ndxi),
                                                 Tlo * np.ones(ndxi)])
                        Thiset[:, :] = np.array([np.zeros(ndxi),
                                                 Thi * np.ones(ndxi)])
                if PS2mfile:
                    PSset[:, :] = np.array([np.zeros(K.shape),
                                            PS.mean(axis=1)])
                if P2mfile:
                    PTset[:, :] = np.array([K, P[:, -1, :].sum(axis=1)])
                    PHset[:, :] = np.array([K, P[:, -1, 0]])
                    PRset[:, :] = np.array([np.zeros(K.shape), P[:, -1, 1]])
                    PPset[:, :] = np.array([np.zeros(K.shape), P[:, -1, 2]])
                    PBset[:, :] = np.array([np.zeros(K.shape), P[:, -1, 3]])
                if G2mfile:
                    Gset[:, :] = np.array([np.zeros(K.shape), G.sum(axis=1)])
                if M2mfile:
                    DCset[:, :] = np.array([dc0 * K, dc])
                    HCset[:, :] = np.array([hc0 * K, hc])
                    BCset[:, :] = np.array([bc0 * K, bc])
                    TCset[:, :] = np.array([tc0 * K, tc])
                    ACset[:, :] = np.array([ac0 * K, ac])
            else:
                mset = Dataset(mapfilef, mode='a')
                # append data
                mset['time'][:] = np.append(mset['time'][:], years[i])
                if U2mfile:
                    mset['ucm'][-1, :] = ucm
                if T2mfile:
                    if tbl:
                        mset['Tc'][-1, :] = Tc[:, -1]
                        mset['Tlo'][-1, :] = Tlo
                        mset['Thi'][-1, :] = Thi
                    else:
                        mset['Tc'][-1, :] = Tc[-1]
                        mset['Tlo'][-1, :] = Tlo * np.ones(ndxi)
                        mset['Thi'][-1, :] = Thi * np.ones(ndxi)
                if PS2mfile:
                    mset['PS'][-1, :] = PS.mean(axis=1)
                if P2mfile:
                    mset['PT'][-1, :] = P[:, -1, :].sum(axis=1)
                    mset['PH'][-1, :] = P[:, -1, 0]
                    mset['PR'][-1, :] = P[:, -1, 1]
                    mset['PP'][-1, :] = P[:, -1, 2]
                    mset['PB'][-1, :] = P[:, -1, 3]
                if G2mfile:
                    mset['G'][-1, :] = G.sum(axis=1)
                if M2mfile:
                    mset['dc'][-1, :] = dc
                    mset['hc'][-1, :] = hc
                    mset['bc'][-1, :] = bc
                    mset['tc'][-1, :] = tc
                    mset['ac'][-1, :] = ac
            mset.close()

        # # his-file
        pbar.set_postfix(inner_loop='write file - his')
        files = [U2hfile, T2hfile, PS2hfile, P2hfile, G2hfile, M2hfile]
        if any(files):
            date0 = datetime.datetime(2000, 1, 1)
            if i == 0:
                hset = Dataset(hisfilef, 'w', format='NETCDF4')
                hset.description = 'Historic simulation data of the coral_model'

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
                if U2hfile:
                    Uset = hset.createVariable('ucm', 'f8',
                                               ('time', 'stations'))
                    Uset.long_name = 'in-canopy flow'
                    Uset.units = 'm s^-1'
                if T2hfile:
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
                if PS2hfile:
                    PSset = hset.createVariable('PS', 'f8',
                                                ('time', 'stations'))
                    PSset.long_name = 'photosynthesis'
                    PSset.units = '-'
                if P2hfile:
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
                if G2hfile:
                    Gset = hset.createVariable('G', 'f8',
                                               ('time', 'stations'))
                    Gset.long_name = 'calcification'
                    Gset.units = 'kg m^-2 d^-1'
                if M2hfile:
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

                # data indices
                xs = xyn['x'].values
                ys = xyn['y'].values
                idx = np.zeros(len(xs))
                for s in range(len(xs)):
                    idx[s] = ((xzw - xs[s]) ** 2 + (yzw - ys[s]) ** 2).argmin()
                idx = idx.astype(int)

                # data
                idates = dates[
                        dates.dt.year == years[i]].reset_index(drop=True)
                t[:] = (idates - date0).dt.days.values
                x[:] = xs
                y[:] = ys
                if U2hfile:
                    Uset[:, :] = np.tile(ucm, (len(idates), 1))[:, idx]
                if T2hfile:
                    if tbl:
                        Tset[:, :] = Tc[idx, :].transpose()
                        Tloset[:, :] = np.tile(Tlo, (len(idates), 1))[:, idx]
                        Thiset[:, :] = np.tile(Thi, (len(idates), 1))[:, idx]
                    else:
                        Tset[:, :] = np.tile(Tc[:], (len(idx), 1)).transpose()
                        Tloset[:, :] = Tlo * np.ones((len(idates), len(idx)))
                        Thiset[:, :] = Thi * np.ones((len(idates), len(idx)))
                if PS2hfile:
                    PSset[:, :] = PS[idx, :].transpose()
                if P2hfile:
                    PTset[:, :] = P[idx, :, :].sum(axis=2).transpose()
                    PHset[:, :] = P[idx, :, 0].transpose()
                    PRset[:, :] = P[idx, :, 1].transpose()
                    PPset[:, :] = P[idx, :, 2].transpose()
                    PBset[:, :] = P[idx, :, 3].transpose()
                if G2hfile:
                    Gset[:, :] = G[idx, :].transpose()
                if M2hfile:
                    DCset[:, :] = np.tile(dc, (len(idates), 1))[:, idx]
                    HCset[:, :] = np.tile(hc, (len(idates), 1))[:, idx]
                    BCset[:, :] = np.tile(bc, (len(idates), 1))[:, idx]
                    TCset[:, :] = np.tile(tc, (len(idates), 1))[:, idx]
                    ACset[:, :] = np.tile(ac, (len(idates), 1))[:, idx]
            else:
                hset = Dataset(hisfilef, mode='a')
                # date conversion
                idates = dates[
                        dates.dt.year == years[i]].reset_index(drop=True)
                t = (idates - date0).dt.days.values
                # append data
                hset['time'][:] = np.append(hset['time'][:], t)
                if U2hfile:
                    hset['ucm'][t, :] = np.tile(ucm, (len(idates), 1))[:, idx]
                if T2hfile:
                    if tbl:
                        hset['Tc'][t, :] = Tc[idx, :].transpose()
                        hset['Tlo'][t, :] = np.tile(
                                Tlo, (len(idates), 1))[:, idx]
                        hset['Thi'][t, :] = np.tile(
                                Thi, (len(idates), 1))[:, idx]
                    else:
                        hset['Tc'][t, :] = np.tile(
                                Tc[:], (len(idx), 1)).transpose()
                        hset['Tlo'][t, :] = Tlo * np.ones(
                                (len(idates), len(idx)))
                        hset['Thi'][t, :] = Thi * np.ones(
                                (len(idates), len(idx)))
                if PS2hfile:
                    hset['PS'][t, :] = PS[idx, :].transpose()
                if P2hfile:
                    hset['PT'][t, :] = P[idx, :, :].sum(axis=2).transpose()
                    hset['PH'][t, :] = P[idx, :, 0].transpose()
                    hset['PR'][t, :] = P[idx, :, 1].transpose()
                    hset['PP'][t, :] = P[idx, :, 2].transpose()
                    hset['PB'][t, :] = P[idx, :, 3].transpose()
                if G2hfile:
                    hset['G'][t, :] = G[idx, :].transpose()
                if M2hfile:
                    hset['dc'][t, :] = np.tile(dc, (len(idates), 1))[:, idx]
                    hset['hc'][t, :] = np.tile(hc, (len(idates), 1))[:, idx]
                    hset['bc'][t, :] = np.tile(bc, (len(idates), 1))[:, idx]
                    hset['tc'][t, :] = np.tile(tc, (len(idates), 1))[:, idx]
                    hset['ac'][t, :] = np.tile(ac, (len(idates), 1))[:, idx]
            hset.close()

# =============================================================================
# # # # finalize the model
# =============================================================================
modelDIMR.finalize()
print('\nModel finalized.')
print('\nEnd time   : {0}'.format(datetime.datetime.now().time()))
