from sys import argv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib as mpl
from netCDF4 import Dataset
from pandas import DataFrame

mapfile = nc.Dataset(
    r"c:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\Zuidgors_bigger\output_Spart_Pucci\VegModel_Puccinellia_map.nc",
    "r",
)

bl = ma.MaskedArray.filled((mapfile.variables["bl"][0, :]), 0.0)  # Bed level
frac_j = ma.MaskedArray.filled((mapfile.variables["veg_frac_j"][:, :, -1]), 0.0)  # Bed level
frac_m = ma.MaskedArray.filled((mapfile.variables["veg_frac_m"][:, :, -1]), 0.0)  # Bed level
veg_frac_j = np.zeros(frac_j.shape)
veg_frac_m = np.zeros(frac_m.shape)
height_j = np.zeros(frac_j.shape)
height_m = np.zeros(frac_m.shape)

dia_j = np.zeros(frac_j.shape)
dia_m = np.zeros(frac_m.shape)

age_j = np.zeros(frac_j.shape)
age_m = np.zeros(frac_m.shape)
root_len_j = np.zeros(frac_j.shape)
root_len_m = np.zeros(frac_m.shape)



veg_frac_m[:, 0][bl>2.5] = 0.9
veg_frac_m[:, 0][bl>3] = 0
height_m[:, 0][bl>= 2.5] = 0.2 # max H winter
height_m[:, 0][bl> 3] = 0
dia_m[:, 0][bl>=2.5] = 0.005
dia_m[:, 0][bl>3] = 0
age_m[:, 0][bl>=2.5] = 10
age_m[:, 0][bl>3] = 0
root_len_m[:, 0][bl>=2.5] = 0.15
root_len_m[:, 0][bl>3] = 0

with Dataset(r"c:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\Zuidgors_bigger\input\VegModel_Puccinellia_map.nc", mode="a") as _map_data:

    _map_data["veg_frac_j"][:, :, -1] = veg_frac_j[:, :]
    _map_data["veg_frac_m"][:, :, -1] = veg_frac_m[:, :]
    _map_data["veg_stemdia_j"][:, :, -1] = dia_j[:, :]
    _map_data["veg_stemdia_m"][:, :, -1] = dia_m[:, :]
    _map_data["veg_height_j"][:, :, -1] = height_j[:, :]
    _map_data["veg_height_m"][:, :, -1] = height_m[:, :]
    _map_data["root_len_j"][:, :, -1] = root_len_j[:, :]
    _map_data["root_len_m"][:, :, -1] = root_len_m[:, :]
    _map_data["veg_age_j"][:, :, -1] = age_j[:, :]
    _map_data["veg_age_m"][:, :, -1] = age_m[:, :]




# _map_data.to_netcdf(path=r"c:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\Zuidgors_bigger\input\input_cover.nc")
# _map_data.close()
# print ('finished saving')