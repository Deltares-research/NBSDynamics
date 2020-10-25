"""
CoralModel v3 - environment

@author: Gijs G. Hendrickx
"""

import os

import pandas as pd
# TODO: consider including the Constants- and Processes-classes to be coded in this file as well instead of utils.py


class Environment:

    def __init__(self, light=None, light_attenuation=None, temperature=None, acidity=None, storm_category=None):
        self.light = light
        self.light_attenuation = light_attenuation
        self.temp = temperature
        self.acid = acidity
        self.storm_category = storm_category

    @property
    def temp_kelvin(self):
        """Temperature in Kelvin."""
        if all(self.temp) < 100.:
            return self.temp + 273.15
        else:
            return self.temp

    @property
    def temp_celsius(self):
        """Temperature in Celsius."""
        if all(self.temp) > 100.:
            return self.temp - 273.15
        else:
            return self.temp

    @property
    def temp_mmm(self):
        monthly_mean = self.temp_kelvin.groupby([
            self.temp_kelvin.index.year, self.temp_kelvin.index.month
        ]).agg(['mean'])
        monthly_maximum_mean = monthly_mean.groupby(level=0).agg(['min', 'max'])
        monthly_maximum_mean.columns = monthly_maximum_mean.columns.droplevel([0, 1])
        return monthly_maximum_mean

    @property
    def dates(self):
        d = self.temp.reset_index().drop('sst', axis=1)
        return pd.to_datetime(d['date'])

    def from_file(self, param, file, file_dir=None):

        def date2index(parameter):
            """Function applicable to time-series in Pandas."""
            parameter['date'] = pd.to_datetime(parameter['date'])
            parameter.set_index('date', inplace=True)

        if file_dir is None:
            f = file
        else:
            f = os.path.join(file_dir, file)

        if param == 'light':
            self.light = pd.read_csv(f, sep='\t')
            date2index(self.light)
        elif param == 'LAC':
            self.light_attenuation = pd.read_csv(f, sep='\t')
            date2index(self.light_attenuation)
        elif param == 'temperature':
            self.temp = pd.read_csv(f, sep='\t')
            date2index(self.temp)
        elif param == 'acidity':
            self.acid = pd.read_csv(f, sep='\t')
            date2index(self.acid)
        elif param == 'storm':
            self.storm_category = pd.read_csv(f, sep='\t')
            self.storm_category.set_index('year', inplace=True)
