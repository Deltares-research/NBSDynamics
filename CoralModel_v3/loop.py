"""
CoralModel v3 - loop

@author: Gijs G. Hendrickx
"""

import numpy as np

from CoralModel_v3 import core, utils


spacetime = (4, 10)
core.RESHAPE = utils.DataReshape(spacetime)

I0 = np.ones(10)
Kd = np.ones(10)
h = np.ones(4)


lme = core.Light(I0, Kd, h)

print(lme.I0.shape)
