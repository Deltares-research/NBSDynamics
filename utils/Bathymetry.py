# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:40:12 2020

@author: hendrick
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# # plot bathymetry
plot = True

# =============================================================================
# # # # file specifications
# =============================================================================
# # old file
old = '40x120.xyz'
old_model = 'FLOW+WAVE'
oldf = os.path.join('p:\\11202744-008-vegetation-modelling', 'students',
                    'GijsHendrickx', 'models', old_model, 'fm', old)

# # new file
new = 'fringing.xyz'
new_model = old_model
newf = os.path.join('p:\\11202744-008-vegetation-modelling', 'students',
                    'GijsHendrickx', 'models', new_model, 'fm', new)

# =============================================================================
# # # # define batymetry function
# =============================================================================


def fringing(y, hdef, ydef, zaxis_down=True):
    # # input check
    # minimum of two depth-points
    if isinstance(hdef, float) or isinstance(ydef, float):
        raise ValueError(
                'More than one depth-point needed to make fringing profile. '
                'Check input of [hdef] and [ydef].')
    # [hdef] and [ydef] must have same length
    if not len(hdef) == len(ydef):
        raise ValueError(
                'Depth-point definitions must have same length. '
                'Check input of [hdef] and [ydef].')

    # # calculations
    try:
        for i in range(len(hdef) - 1):
            if y > ydef[i] and y <= ydef[i + 1]:
                h = ((hdef[i] - hdef[i + 1]) / (ydef[i] - ydef[i + 1]) *
                     (y - ydef[i]))
    except ValueError:
        h = np.zeros(len(y))
        for i in range(len(y)):
            for j in range(len(ydef) - 1):
                if y[i] > ydef[j] and y[i] <= ydef[j + 1]:
                    h[i] = ((hdef[j] - hdef[j + 1]) / (ydef[j] - ydef[j + 1]) *
                            (y[i] - ydef[j])) + hdef[j]

    # # output
    if not zaxis_down:
        h = -h

    return h


# =============================================================================
# # # # write bathymetry file
# =============================================================================
# define profile
hdef = np.array([50., 50., 20., 3., 2., -2.])
ydef = np.array([100., 200., 250., 450., 700., 750.])
# load location data points
xyz = pd.read_csv(oldf, delim_whitespace=True,
                  header=None, names=['x', 'y', 'z'])
# rewrite z-coordinates
xyz['z'] = fringing(xyz['y'], hdef, ydef, zaxis_down=False)
# write to .xyz-file
xyz.to_csv(newf, header=None, index=None, sep='\t', mode='w')

# =============================================================================
# # # # plot bathymetry
# =============================================================================
if plot:
    fig, ax = plt.subplots()
    # zero-line
    ax.plot([ydef.min(), ydef.max()], [0., 0.],
            color='gray', alpha=.5,
            linestyle='solid', linewidth=.5,
            label='_nolegend_')
    # plot data
    ax.plot(ydef, -hdef,
            color='black', alpha=1.,
            linestyle='solid', linewidth=1.,
            label='_nolegend_')
    # axes labels
    ax.set_xlabel(r'y-coodinate [m]')
    ax.set_ylabel(r'water depth [m]')
    # axes ticks
    # plot limits
    ax.set_xlim([150., 750.])
    # explaining lines / texts / etc.
    # legend / title
