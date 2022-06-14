import numpy as np

from src.biota_models.mangroves.model.mangrove_model import Mangrove

class Hydro_Morphodynamics:
    """Get the values for the hydromorphodynamic environment"""

    def __init__(self, tau_cur, u_cur, wl_cur, bl_cur,ba, ts, mangrove: Mangrove):

        self.tau = tau_cur
        self.u = u_cur
        self.wl = wl_cur
        self.bl = bl_cur
        self.ts = ts
        self.ba = ba
        if ts == 0:
            mangrove.tau_ts = self.tau
            mangrove.u_ts = self.u
            mangrove.wl_ts = self.wl
            mangrove.bl_ts = self.bl
            mangrove.ba_ts = self.ba
        else:
            mangrove.tau_ts = np.column_stack((mangrove.tau_ts, self.tau))
            mangrove.u_ts = np.column_stack((mangrove.u_ts, self.u))
            mangrove.wl_ts = np.column_stack((mangrove.wl_ts, self.wl))
            mangrove.bl_ts = np.column_stack((mangrove.bl_ts, self.bl))
            mangrove.ba_ts = np.column_stack((mangrove.ba_ts, self.ba))

    def get_hydromorph_values(self, mangrove):
        mangrove.max_tau = np.zeros(len(mangrove.tau_ts))
        mangrove.max_u = np.zeros(len(mangrove.u_ts))
        mangrove.max_wl = np.zeros(len(mangrove.wl_ts))
        mangrove.min_wl = np.zeros(len(mangrove.wl_ts))
        mangrove.bl = np.zeros(len(mangrove.bl_ts))
        mangrove.ba = np.zeros(len(mangrove.bl_ts))

        for i in range(0, len(mangrove.tau_ts)):
            mangrove.max_tau[i] = max(mangrove.tau_ts[i, :])
            mangrove.max_u[i] = max(mangrove.u_ts[i, :])
            mangrove.max_wl[i] = max(mangrove.wl_ts[i, :])
            mangrove.min_wl[i] = min(mangrove.wl_ts[i, :])
        mangrove.bl[:] = mangrove.bl_ts[:, -1]  # last values in bed level to get 'current' value
        mangrove.ba[:] = mangrove.ba_ts[:, -1]  # last values in grid area to get 'current' value

        flood = np.zeros(mangrove.wl_ts.shape)
        flood[mangrove.wl_ts > mangrove.constants.fl_dr] = 1
        flood_count = np.zeros(len(mangrove.wl_ts))
        for i in range(0, len(flood_count)):
            flood_count[i] = np.count_nonzero(flood[i, :])
        mangrove.inun_rel = flood_count / len(mangrove.wl_ts[0, :])

    def store_hydromorph_values(self, mangrove):
        mangrove.max_tau_prev = mangrove.max_tau
        mangrove.max_u_prev = mangrove.max_u
        mangrove.max_wl_prev = mangrove.max_wl
        mangrove.min_wl_prev = mangrove.min_wl
        mangrove.bl_prev = mangrove.bl
        mangrove.wl_prev = mangrove.wl_ts
