from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import root_validator
import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import netCDF4 as nc
import numpy.ma as ma


from src.biota_models.mangroves.model.mangrove_constants import MangroveConstants

from src.core.biota.biota_model import Biota
from src.core.common.singletons import RESHAPE

MangroveAttribute = Union[float, list, tuple, np.ndarray]


class Mangrove(Biota):
    """
    Implements the `MangroveProtocol`.
    Mangrove object, representing one plant.
    """

    constants: Optional[MangroveConstants]
    total_cover: Optional[
       MangroveAttribute
    ]  # sum of fraction of area coverage in each cell (for all ages)
    mort: Optional[MangroveAttribute] = None

    # Mangrove characteristics
    stem_num: Optional[MangroveAttribute] = None
    stem_dia: Optional[MangroveAttribute] = None
    height: Optional[MangroveAttribute] = None
    density: Optional[MangroveAttribute] = None
    age: Optional[MangroveAttribute] = None
    density: Optional[MangroveAttribute] = None
    root_dia: Optional[MangroveAttribute] = None
    root_height: Optional[MangroveAttribute] = None
    root_density: Optional[MangroveAttribute] = None



    # hydromorphodynamic environment
    max_tau: Optional[MangroveAttribute] = None
    max_u: Optional[MangroveAttribute] = None
    max_wl: Optional[MangroveAttribute] = None
    min_wl: Optional[MangroveAttribute] = None
    bl: Optional[MangroveAttribute] = None
    max_tau_prev: Optional[MangroveAttribute] = None
    max_u_prev: Optional[MangroveAttribute] = None
    max_wl_prev: Optional[MangroveAttribute] = None
    min_wl_prev: Optional[MangroveAttribute] = None
    bl_prev: Optional[MangroveAttribute] = None
    wl_prev: Optional[MangroveAttribute] = None
    tau_ts: Optional[MangroveAttribute] = None
    u_ts: Optional[MangroveAttribute] = None
    wl_ts: Optional[MangroveAttribute] = None
    bl_ts: Optional[MangroveAttribute] = None
    inun_rel: Optional[MangroveAttribute] = None

    def __repr__(self):
        """Development representation."""
        return f"Characteristics({self.height}, {self.stem_dia}, {self.age}, {self.density})"

    def __str__(self):
        """Print representation."""
        return (
            f"Vegetation characteristics with: height = {self.height} m; stem_dia = {self.stem_dia} m; age = {self.age} days; "
            f"density = {self.density}"
        )

    @property
    def total_cover(self):  # as input for DFM
        # take cover as sum of all the ages and life stages
        # self.total_cover = self.juvenile.cover + self.mature.cover
        return self.juvenile.cover + self.mature.cover

    @property
    def veg_den(self):  # as input for DFM
        """stem density in number of stems per m2, according to area fraction of veg age"""
        return (self.juvenile.stem_num * self.juvenile.veg_frac).sum(axis=1) + (
                self.mature.stem_num * self.mature.veg_frac
        ).sum(axis=1)

    @property
    def av_stemdia(self):  # as input for DFM
        """average stem diameter of the different vegetation in one grid cell"""
        cover_j = self.juvenile.cover.copy()
        cover_m = self.mature.cover.copy()
        cover_j[cover_j == 0] = 1
        cover_m[cover_m == 0] = 1
        # if self.juvenile.cover.all() == 0 and self.mature.cover.all() == 0:
        #     return np.zeros(self.cover.shape)
        # elif self.mature.cover.all() == 0:
        #     return (self.juvenile.stem_dia * self.juvenile.veg_frac).sum(axis=1) / self.juvenile.cover
        # elif self.juvenile.cover.all() == 0:
        #     return (self.mature.stem_dia * self.mature.veg_frac).sum(axis=1) / self.mature.cover
        # else:
        return (self.juvenile.stem_dia * self.juvenile.veg_frac).sum(axis=1).reshape(
            -1, 1
        ) / cover_j + (self.mature.stem_dia * self.mature.veg_frac).sum(axis=1).reshape(
            -1, 1
        ) / cover_m

    @property
    def av_height(self):  # as input for DFM
        """average shoot height of the different vegetation in one grid cell"""
        cover_j = self.juvenile.cover.copy()
        cover_m = self.mature.cover.copy()
        cover_j[cover_j == 0] = 1
        cover_m[cover_m == 0] = 1
        # if np.all(self.juvenile.cover == 0) and np.all(self.mature.cover == 0):
        #     return np.zeros(self.cover.shape)
        # elif np.all(self.mature.cover == 0):
        #     return (self.juvenile.veg_height * self.juvenile.veg_frac).sum(axis=1) / (self.juvenile.cover[self.juvenile.cover == 0]=1)
        # elif self.juvenile.cover.all() == 0:
        #     return (self.mature.veg_height * self.mature.veg_frac).sum(axis=1) / self.mature.cover
        # else:
        return (self.juvenile.veg_height * self.juvenile.veg_frac).sum(axis=1).reshape(
            -1, 1
        ) / cover_j + (self.mature.veg_height * self.mature.veg_frac).sum(
            axis=1
        ).reshape(
            -1, 1
        ) / cover_m

    def initiate_mangrove_characteristics(self, cover: Optional[Path]):

        self.mort = np.zeros(_reshape.space)

        if not cover:
            self.stem_dia = np.zeros(_reshape.space)
            self.stem_dia = self.stem_dia.reshape(len(self.stem_dia), 1)
            self.stem_num = np.zeros(self.stem_dia.shape)
            self.height = np.zeros(self.stem_dia.shape)
            self.age = np.zeros(self.stem_dia.shape)
            self.density = np.zeros(self.stem_dia.shape)
            self.root_height = np.zeros(self.stem_dia.shape)
            self.root_dia = np.zeros(self.stem_dia.shape)
            self.root_density = np.zeros(self.stem_dia.shape)

        else:
            input_cover: dict = nc.Dataset(cover, "r")
            self.stem_dia = ma.MaskedArray.filled(
                (input_cover.variables["stem_dia"][:, :, -1]), 0.0
            )
            self.height =ma.MaskedArray.filled(
                (input_cover.variables["height"][:, :, -1]), 0.0
            )
            self.age =ma.MaskedArray.filled(
                (input_cover.variables["age"][:, :, -1]), 0.0
            )
            self.cover = ma.MaskedArray.filled(
                (input_cover.variables["cover"][:, :, -1]), 0.0
            )


    def update_mangrove_characteristics(self):


