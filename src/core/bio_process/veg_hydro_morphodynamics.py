import numpy as np
from scipy.optimize import newton

from src.core import RESHAPE
from src.core.common.constants import Constants

class Hydro_Morphodynamics:
    """hydromorphodynamic environment"""

    def __init__(
            self,
            tau_cur,
            u_cur,
            wl_cur,
            bl_cur,
            constants: Constants = Constants(),
    ):
            if self.tau is None:
                self.tau = tau_cur
                self.u = u_cur
                self.wl = wl_cur
                self.bl = bl_cur
            else:
                self.tau = np.column_stack((self.tau, tau_cur))
                self.u = np.column_stack((self.u, u_cur))
                self.wl = np.column_stack((self.wl, wl_cur))
                self.bl = np.column_stack((self.bl, bl_cur))

    def get_hydromorph_values(self, veg):
        veg.max_tau = np.zeros(len(self.tau))
        veg.max_u = np.zeros(len(self.u))
        veg.max_wl = np.zeros(len(self.wl))
        veg.min_wl = np.zeros(len(self.wl))
        veg.bl = np.zeros(len(self.bl))

        for i in range(len(self.wl)):
            veg.max_tau[i] = max(self.tau[i, :])
            veg.max_u[i] = max(self.u[i, :])
            veg.max_wl[i] = max(self.wl[i, :])
            veg.min_wl[i] = min(self.wl[i, :])
        veg.bl[:] = self.wl[:, -1] #last values in bed level to get 'current' value

    def clean_hyromorph_matrixes(self):
        self.tau = []
        self.u = []
        self.wl = []
        self.bl = []



