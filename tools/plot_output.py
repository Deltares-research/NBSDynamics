# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:36:48 2021

@author: herman
"""

import netCDF4
import os
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')

# NetCDF4-Python can read a remote OPeNDAP dataset or a local NetCDF file:
out_dir = os.path.join('C:\\','Users','herman', 'OneDrive - Stichting Deltares',
                        'Documents','PyProjects','Mariya_model','Run_Transect','output')

# read map file and plot
url=os.path.join(out_dir,'CoralModel_map.nc')
nc = netCDF4.Dataset(url)
nc.variables.keys()

teller = 0
for vv in nc.variables.keys():
    teller = teller+1
    if(teller >3):
       
        VT = nc.variables[vv]
        VarT = VT[:]

        fig = plt.figure()
        ax = plt.axes()
        plt.xlim(0, 100)
        plt.ylim(0, 1.05*np.max(VarT))
        plt.title(VT.long_name)
        plt.xlabel("Time (years)")
        plt.ylabel(VT.units)
        
        x = np.linspace(1,100,100)
        
        ax.plot(x, VarT[:,1],'-g',label='Cell 1')
        ax.plot(x, VarT[:,100],'-r',label="Cell 100")
        ax.plot(x,VarT[:,300],'-c',label='Cell 300')
        plt.legend();

nc.close()

# read his file and plot
url=os.path.join(out_dir,'CoralModel_his.nc')
nc = netCDF4.Dataset(url)
nc.variables.keys()

teller = 0
for vv in nc.variables.keys():
    teller = teller+1
    if(teller >3):
       
        VT = nc.variables[vv]
        VarT = VT[:]

        fig = plt.figure()
        ax = plt.axes()
        plt.xlim(0, 100)
        plt.ylim(0, 1.05*np.max(VarT))
        plt.title(VT.long_name)
        plt.xlabel("Time (years)")
        plt.ylabel(VT.units)
        
        x = np.linspace(0,100,36525)
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, VarT.shape[1])))
        for i in range(VarT.shape[1]):
            ax.plot(x, VarT[:,i],color=next(colors),label=f'Point {i}')
        plt.legend();

nc.close()
