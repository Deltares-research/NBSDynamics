import numpy as np
from src.core.vegetation.veg_model import Vegetation


class Hydro_Morphodynamics:
    """hydromorphodynamic environment"""

    def __init__(
            self,
            tau_cur,
            u_cur,
            wl_cur,
            bl_cur,
            ts,
            veg: Vegetation
    ):

        self.tau = tau_cur
        self.u = u_cur
        self.wl = wl_cur
        self.bl = bl_cur
        self.ts = ts
        if ts == 0:
            veg.tau_ts = self.tau
            veg.u_ts = self.u
            veg.wl_ts = self.wl
            veg.bl_ts = self.bl
        else:
            veg.tau_ts = np.column_stack((veg.tau_ts, self.tau))
            veg.u_ts = np.column_stack((veg.u_ts, self.u))
            veg.wl_ts = np.column_stack((veg.wl_ts, self.wl))
            veg.bl_ts = np.column_stack((veg.bl_ts, self.bl))

    def get_hydromorph_values(self, veg):
        veg.max_tau = np.zeros(len(veg.tau_ts))
        veg.max_u = np.zeros(len(veg.u_ts))
        veg.max_wl = np.zeros(len(veg.wl_ts))
        veg.min_wl = np.zeros(len(veg.wl_ts))
        veg.bl = np.zeros(len(veg.bl_ts))

        for i in range(0, len(veg.tau_ts)):
            veg.max_tau[i] = max(veg.tau_ts[i, :])
            veg.max_u[i] = max(veg.u_ts[i, :])
            veg.max_wl[i] = max(veg.wl_ts[i, :])
            veg.min_wl[i] = min(veg.wl_ts[i, :])
        veg.bl[:] = veg.bl_ts[:, -1]  # last values in bed level to get 'current' value

    def store_hydromorph_values(self, veg):
        veg.max_tau_prev = veg.max_tau
        veg.max_u_prev = veg.max_u
        veg.max_wl_prev = veg.max_wl
        veg.min_wl_prev = veg.min_wl
        veg.bl_prev = veg.bl
        veg.wl_prev = veg.wl_ts

