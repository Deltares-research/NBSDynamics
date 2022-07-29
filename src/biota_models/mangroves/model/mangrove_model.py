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
    root_num: Optional[MangroveAttribute] = None




    # hydromorphodynamic environment
    max_tau: Optional[MangroveAttribute] = None
    max_u: Optional[MangroveAttribute] = None
    max_wl: Optional[MangroveAttribute] = None
    min_wl: Optional[MangroveAttribute] = None
    bl: Optional[MangroveAttribute] = None
    ba: Optional[MangroveAttribute] = None
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
    ba_ts: Optional[MangroveAttribute] = None
    inun_rel: Optional[MangroveAttribute] = None
    bio_total_cell: Optional[MangroveAttribute] = None
    I: Optional[MangroveAttribute] = None
    C: Optional[MangroveAttribute] = None

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
    def veg_den(self):  # as input for DFM
        """stem density in number of stems per m2, according to area fraction of veg age"""
        return (self.stem_num + self.root_num) / self.ba[:, None] ## stem number divided by grid cell size [N/m2]

    @property
    def av_stemdia(self):  # as input for DFM
        """average stem diameter of mangroves in one grid cell"""
        av_dia = np.sum(self.stem_num * self.stem_dia, axis=1)/ np.sum(self.stem_num, axis=1) + np.sum(self.root_num * self.constants.root_dia, axis=1)/np.sum(self.root_num, axis=1)
        av_dia[np.isnan(av_dia)] = 0
        return av_dia

    @property
    def av_height(self):  # as input for DFM
        """average shoot height of mangroves in one grid cell"""
        av_h = np.sum(self.stem_num * self.height, axis=1)/ np.sum(self.stem_num, axis=1) + np.sum(self.root_num * self.constants.root_height, axis=1)/np.sum(self.root_num, axis=1)
        av_h[np.isnan(av_h)] = 0
        return av_h

    @property
    def bio_total_cell(self):
        return np.sum((self.stem_num * (self.constants.bio_a*self.stem_dia**self.constants.ind_a + self.constants.bio_b*self.stem_dia**self.constants.ind_b)), axis=1)

    @property
    def B_05(self):
        W_mature = self.constants.bio_a*self.constants.MaxD**self.constants.ind_a + self.constants.bio_b*self.constants.MaxD**self.constants.ind_b
        R = 10*np.sqrt(self.constants.MaxD/(2*100))
        return self.ba/((2*R)**2)*W_mature

    def initiate_mangrove_characteristics(self, cover: Optional[Path]):
        _reshape = RESHAPE()
        self.mort = np.zeros(_reshape.space)

        if not cover:
            self.stem_dia = np.zeros(_reshape.space)
            self.stem_dia = self.stem_dia.reshape(len(self.stem_dia), 1)
            self.stem_num = np.zeros(self.stem_dia.shape)
            self.height = np.zeros(self.stem_dia.shape)
            self.root_num = np.zeros(self.stem_dia.shape)
            self.ba = np.zeros(self.stem_dia.shape)
        else:
            input_cover: dict = nc.Dataset(cover, "r")
            self.stem_dia = ma.MaskedArray.filled(
                (input_cover.variables["stem_dia"][:, :, -1]), 0.0
            )
            self.height =ma.MaskedArray.filled(
                (input_cover.variables["height"][:, :, -1]), 0.0
            )
            self.stem_num =ma.MaskedArray.filled(
                (input_cover.variables["stem_num"][:, :, -1]), 0.0
            )
            self.root_num = ma.MaskedArray.filled(
                (input_cover.variables["root_num"][:, :, -1]), 0.0
            )
            self.bio_total_cell = ma.MaskedArray.filled(
                (input_cover.variables["bio_total_cell"][:, :, -1]), 0.0
            )


    def update_mangrove_characteristics(self, col: [Optional] = None,  stem_dia: [Optional] = None):
        if not col:
            self.stem_dia[self.stem_num == 0] = 0 #account for dieback
            self.stem_dia = self.stem_dia + ((self.constants.G*self.stem_dia*(1-self.stem_dia*(self.height*100)/(self.constants.MaxD*self.constants.MaxH))*(self.I[:, -1]*self.C[:]).reshape(-1, 1))/(274+3*self.constants.b2*self.stem_dia-4*self.constants.b3*self.stem_dia**2)*12)

            self.height = (137+self.constants.b2*self.stem_dia-self.constants.b3*self.stem_dia**2)/100
            self.height[self.stem_num == 0] = 0

            self.root_num = self.constants.m*(1/(1+ np.exp(self.constants.f*(self.constants.MaxD/2 - self.stem_dia))))
            self.root_num[self.stem_num == 0] = 0
        else:
            self.height[:, 0][stem_dia[:, 0]>0] = (137 + self.constants.b2 * stem_dia[:, 0][stem_dia[:, 0]>0] - self.constants.b3 * stem_dia[:, 0][stem_dia[:, 0]>0] ** 2) / 100
            self.root_num[:, 0][stem_dia[:, 0]>0] = self.constants.m * (
                        1 / (1 + np.exp(self.constants.f * (self.constants.MaxD / 2 - stem_dia[:, 0][stem_dia[:, 0]>0]))))




