"""
CoralModel v3 - loop

@author: Gijs G. Hendrickx
"""

import numpy as np

from CoralModel_v3 import core, utils, environment, hydrodynamics


# TODO: Write the model execution as a function to be called in "interface.py".
# TODO: Include a model execution in which all processes can be switched on and off; based on Processes. This also
#  includes the main environmental factors, etc.

spacetime = (4, 10)
core.RESHAPE = utils.DataReshape(spacetime)

I0 = np.ones(10)
Kd = np.ones(10)
h = np.ones(4)


lme = core.Light(I0, Kd, h)

print(lme.I0.shape)
