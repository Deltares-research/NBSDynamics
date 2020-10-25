# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:37:17 2020

@author: hendrick
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
from tqdm import tqdm
import datetime
from netCDF4 import Dataset
import faulthandler
faulthandler.enable()

# =============================================================================
# # # # specify directories of ddl- and input-files
# =============================================================================
D3D_HOME = os.path.join('p:\\11202744-008-vegetation-modelling', 'code_1709',
                        'windows', 'oss_artifacts_x64_63721', 'x64')
dflow_dir = os.path.join(D3D_HOME, 'dflowfm', 'bin', 'dflowfm.dll')
dimr_path = os.path.join(D3D_HOME, 'dimr', 'bin', 'dimr_dll.dll')
# work directory
model_folder = 'rt003'
workdir = os.path.join('p:\\11202744-008-vegetation-modelling', 'students',
                       'GijsHendrickx', 'models', 'RunTimeD3D', model_folder)
inputdir = os.path.join(workdir, 'timeseries')
# input files (Delft3D)
config_file = os.path.join(workdir, 'dimr_config.xml')
mdufile = os.path.join(workdir, 'fm', 'FlowFM.mdu')
# print directories and input-files as check
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
# # # # define functions
# =============================================================================


def dc_av(dc, hc, bc, tc):
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
    dc_rep = (bc * (hc - tc) + dc * tc) / hc

    # # output
    return dc_rep


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
    dc_rep = dc_av(dc, hc, bc, tc)
    # representative vegetation density
    rnveg = (2 * dc_rep) / (ac ** 2)

    # # output
    return rnveg


# =============================================================================
# # # # set time parameters for coupled model
# =============================================================================
# # time-span
dperm = 7
# simulation time [days]
T = 4 * dperm

# # model time per vegetation step [s]
mtpervt = np.array([86400,
                    43200, 43200,
                    21600, 21600, 21600, 21600,
                    10800, 10800, 10800, 10800,
                    10800, 10800, 10800, 10800])
mtpervt = np.repeat(mtpervt, int(dperm))
mtpervt = np.append(3600, mtpervt)

# =============================================================================
# # # # define output
# =============================================================================
# # map-file
# map-file directory
mapfile = 'CoralModel_map.nc'
mapfilef = os.path.join(outputfolder, mapfile)
# time-interval > mtpervt

# # his-file
# location(s)
xynfilef = os.path.join(workdir, 'fm', 'FlowFm_obs.xyn')
xyn = pd.read_csv(xynfilef, header=None, delim_whitespace=True)
xyn.columns = ['x', 'y', 'name']
# his-file directory
hisfile = 'CoralModel_his.nc'
hisfilef = os.path.join(outputfolder, hisfile)
# time-interval > mtpervt

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
yvbndmax = ybndmax

# =============================================================================
# # # # initialisation of vegetation variables
# =============================================================================
# # initial morphology
dc0 = .1  # m
hc0 = .2  # m
bc0 = dc0
tc0 = hc0
ac0 = .2  # m

# # carrying capacity
K = np.zeros(ndxi)
K[np.logical_and.reduce((xzw >= xvbndmin,
                         xzw <= xvbndmax,
                         yzw >= yvbndmin,
                         yzw <= yvbndmax))] = 1.

# # morphological dimensions
# diameter plate
dc = K * dc0
# coral height ~ stemheight
hc = modelFM.get_var('stemheight')
hc[range(ndxi)] = K * hc0
modelFM.set_var('stemheight', hc)
# diameter base
bc = K * bc0
# thickness plate
tc = K * tc0
# axial distance
ac = K * ac0
# representative diameter
diaveg = modelFM.get_var('diaveg')
diaveg[range(ndxi)] = K * dc_av(dc0, hc0, bc0, tc0)
modelFM.set_var('diaveg', diaveg)
# 'vegetation' density
rnveg = modelFM.get_var('rnveg')
rnveg[range(ndxi)] = K * morph2vegden(dc0, hc0, bc0, tc0, ac0)
modelFM.set_var('rnveg', rnveg)

# =============================================================================
# # # # run the model
# =============================================================================
print('Start time : {0}\n'.format(datetime.datetime.now().time()))

for i in tqdm(range(len(mtpervt))):
    # # update hydrodynamic model
    modelDIMR.update(mtpervt[i])

    # # extract variables from DFM via BMI
    # flow characteristics
    is_sumvalsnd = modelFM.get_var('is_sumvalsnd')
    is_maxvalsnd = modelFM.get_var('is_maxvalsnd')
    is_dtint = modelFM.get_var('is_dtint')
    uwav = modelFM.get_var('Uorb')
    twav = modelFM.get_var('twav')
    # param[range(ndxi), i]
    # > i = 0 : shear stress   [tau]
    # > i = 1 : flow velocity  [vel]
    # > i = 2 : water depth    [wd]
    # morphological characteristics

    # # calculate (mean and max.) values from DFM data
    vel_mean = is_sumvalsnd[range(ndxi), 1] / is_dtint
    vel_max = is_maxvalsnd[range(ndxi), 1]
    wd_mean = is_sumvalsnd[range(ndxi), 2] / is_dtint

    # # show intermediate model output

    # # return coral data to hydrodynamic model
    # reset counters
    is_sumvalsnd.fill(0.)
    is_maxvalsnd.fill(0.)
    is_dtint.fill(0.)
    # push counters and updated coral field to model
    modelFM.set_var('is_sumvalsnd', is_sumvalsnd)
    modelFM.set_var('is_maxvalsnd', is_maxvalsnd)
    modelFM.set_var('is_dtint', is_dtint)

    # # write model results in file
    # map-file
    if i == 0:
        mset = Dataset(mapfilef, 'w', format='NETCDF4')
        mset.description = 'Mapped simulation data of the RunTimeD3D-models.'

        # dimensions
        mset.createDimension('time', None)
        mset.createDimension('nmesh2d_face', int(ndxi))

        # variables
        dt = mset.createVariable('time', int, ('time',))
        dt.long_name = 'Delft3D-FM run time per vegetation time-step'
        dt.units = 's'

        x = mset.createVariable('mesh2d_x', 'f8', ('nmesh2d_face',))
        x.long_name = 'x-coordinate'
        x.units = 'm'

        y = mset.createVariable('mesh2d_y', 'f8', ('nmesh2d_face',))
        y.long_name = 'y-coordinate'
        y.units = 'm'

        Uwset = mset.createVariable('uw', 'f8',
                                    ('time', 'nmesh2d_face'))
        Uwset.long_name = 'wave orbital velocity'
        Uwset.units = 'm s^-1'

        Ubset = mset.createVariable('ub', 'f8',
                                    ('time', 'nmesh2d_face'))
        Ubset.long_name = 'mean bulk flow velocity'
        Ubset.units = 'm s^-1'

        Umset = mset.createVariable('ubmax', 'f8',
                                    ('time', 'nmesh2d_face'))
        Umset.long_name = 'maximum bulk flow velocity'
        Umset.units = 'm s^-1'

        Hmset = mset.createVariable('h', 'f8',
                                    ('time', 'nmesh2d_face'))
        Hmset.long_name = 'mean water depth'
        Hmset.units = 'm'

        Twset = mset.createVariable('Tp', 'f8',
                                    ('time', 'nmesh2d_face'))
        Twset.long_name = 'peak wave period'
        Twset.units = 's'

        # data
        dt[:] = mtpervt[i]
        x[:] = xzw[range(ndxi)]
        y[:] = yzw[range(ndxi)]
        Uwset[-1, :] = uwav[range(ndxi)]
        Ubset[-1, :] = vel_mean[range(ndxi)]
        Umset[-1, :] = vel_max[range(ndxi)]
        Hmset[-1, :] = wd_mean[range(ndxi)]
        Twset[-1, :] = twav[range(ndxi)]
    else:
        mset = Dataset(mapfilef, mode='a')
        # append data
        mset['time'][:] = np.append(mset['time'][:], mtpervt[i])
        mset['uw'][-1, :] = uwav[range(ndxi)]
        mset['ub'][-1, :] = vel_mean[range(ndxi)]
        mset['ubmax'][-1, :] = vel_max[range(ndxi)]
        mset['h'][-1, :] = wd_mean[range(ndxi)]
        mset['Tp'][-1, :] = twav[range(ndxi)]
    mset.close()
    # his-file
    if i == 0:
        hset = Dataset(hisfilef, 'w', format='NETCDF4')
        hset.description = 'Historic simulation data of the RunTimeD3D-models.'

        # dimensions
        hset.createDimension('time', None)
        hset.createDimension('stations', len(xyn))

        # variables
        dt = hset.createVariable('time', int, ('time',))
        dt.long_name = 'Delft3D-FM run time per vegetation time-step'
        dt.units = 's'

        x = hset.createVariable('station_x_coordinate', 'f8',
                                ('stations',))
        x.long_name = 'x-coordinate of station(s)'
        x.units = 'm'

        y = hset.createVariable('station_y_coordinate', 'f8',
                                ('stations',))
        y.long_name = 'y-coordinate of station(s)'
        y.units = 'm'

        Uwset = hset.createVariable('uw', 'f8',
                                    ('time', 'stations'))
        Uwset.long_name = 'wave orbital velocity'
        Uwset.units = 'm s^-1'

        Ubset = hset.createVariable('ub', 'f8',
                                    ('time', 'stations'))
        Ubset.long_name = 'mean bulk flow velocity'
        Ubset.units = 'm s^-1'

        Umset = hset.createVariable('ubmax', 'f8',
                                    ('time', 'stations'))
        Umset.long_name = 'maximum bulk flow velocity'
        Umset.units = 'm s^-1'

        Hmset = hset.createVariable('h', 'f8',
                                    ('time', 'stations'))
        Hmset.long_name = 'mean water depth'
        Hmset.units = 'm'

        Twset = hset.createVariable('Tp', 'f8',
                                    ('time', 'stations'))
        Twset.long_name = 'peak wave period'
        Twset.units = 's'

        # data indices
        xs = xyn['x'].values
        ys = xyn['y'].values
        idx = np.zeros(len(xs))
        for s in range(len(xs)):
            idx[s] = ((xzw - xs[s]) ** 2 + (yzw - ys[s]) ** 2).argmin()
        idx = idx.astype(int)

        # data
        dt[:] = mtpervt[i]
        x[:] = xs
        y[:] = ys
        Uwset[:, :] = uwav[idx].RESHAPE((1, len(idx)))
        Ubset[:, :] = vel_mean[idx].RESHAPE((1, len(idx)))
        Umset[:, :] = vel_max[idx].RESHAPE((1, len(idx)))
        Hmset[:, :] = wd_mean[idx].RESHAPE((1, len(idx)))
        Twset[:, :] = twav[idx].RESHAPE((1, len(idx)))
    else:
        hset = Dataset(hisfilef, mode='a')
        # append data
        hset['time'][:] = np.append(hset['time'][:], mtpervt[i])
        hset['uw'][-1, :] = uwav[idx].RESHAPE((1, len(idx)))
        hset['ub'][-1, :] = vel_mean[idx].RESHAPE((1, len(idx)))
        hset['ubmax'][-1, :] = vel_max[idx].RESHAPE((1, len(idx)))
        hset['h'][-1, :] = wd_mean[idx].RESHAPE((1, len(idx)))
        hset['Tp'][-1, :] = twav[idx].RESHAPE((1, len(idx)))
    hset.close()

# =============================================================================
# # # # finalize the model run
# =============================================================================
modelDIMR.finalize()
print('\nModel finalized.')
print('\nEnd time   : {0}'.format(datetime.datetime.now().time()))
