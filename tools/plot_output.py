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
url=os.path.join(out_dir,'CoralModel_map.nc')
nc = netCDF4.Dataset(url)
nc.variables.keys()

PT = nc.variables['PT'][:].data
nc.close()


fig = plt.figure()
ax = plt.axes()
plt.xlim(0, 100)
plt.ylim(0, 1.05)
plt.title("Total Coral Population")
plt.xlabel("Time (years)")
plt.ylabel("Relative cover")

x = np.linspace(1,100,100)

ax.plot(x, PT[:,1],'-g',label='Cell 1')
ax.plot(x, PT[:,100],'-r',label="Cell 100")
ax.plot(x,PT[:,300],'-c',label='Cell 300')
plt.legend();