# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:14:12 2020

@author: hendrick
"""

import numpy as np
import pandas as pd
import os
import datetime
import dateutil
import matplotlib.pyplot as plt
import Supporting_func as support

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# =============================================================================
# =============================================================================
# # # # # Function definitions
# =============================================================================
# =============================================================================
# 1. PAR time-series
# 2. SST time-series
# 3. Hydrodynamics time-series
# 4. Aragonite time-series

# =============================================================================
# # # # PAR time-series
# =============================================================================


def TS_PAR(Tstart, Tend, lat, n0=79.25, eps=23.4398, e=.016704, lop=282.895,
           lat_unit='deg', S0=1600., write=False, path=None,
           print_output=True):
    """
    Daily average light-intensity due to the seasons, the latitude, and the
    declination of the sun.

    Parameters
    ----------
    Tstart : datetime.date, string
        Start date given as datetime.date or as string; string formatting is
        str(YYYY-MM-DD) or str(DD-MM-YYYY).
    Tend : datetime.date, string
        End date given as datetime.date or as string; string formatting is
        str(YYYY-MM-DD) or str(DD-MM-YYYY).
    lat : numeric
        Latitude [deg] / [rad].
    n0 : numeric, optional
        Number of day of the year of the vernal equinox (around March 20-21).
        Default: n0 = 79.25 [th day of the year].
    eps : numeric, optional
        Obliquity of the Earth [degrees].
    ecc : numeric, optional
        Eccentricity of the Earth [-].
    lop : numeric, optional
        Longitude of perihelion [degrees].
    lat_unit : string, optional
        Unit in which the latitude is given.
        Options: degrees ['deg'], and radians ['rad'].
    S0 : string, numeric, optional
        Solar constant [W m^-2].
        Options:
        [string('norm')] normalises the daily-averaged solar insolation to
        values between 0 and 1 w.r.t. the Earth as a whole;
        [str('norm_lat')] normalises the daily-averaged solar insolation to
        values between 0 and 1 w.r.t. specified latitude; and
        [numeric] uses the value as specified by the user to be the solar
        constant in [W m^-2] or any other unit; e.g. [umol photons m^-2 s^-1].
    print_output : boolean, optional
        Print the output to a parameter.

    Returns
    -------
    Id : numeric, array
        Daily-averaged insolation [W m^-2]. (Follows the unit as specified for
        the solar constant.)
    """
    # # check input
    # solar constant
    S0s = ['norm', 'norm_lat']
    if isinstance(S0, str):
        if S0 not in S0s:
            raise KeyError(
                    'Invalide input of S0. Expected float, array, integer '
                    'or one of: {0}'.format(S0s))
    elif not isinstance(S0, (float, np.ndarray, int)):
        raise KeyError(
                'Invalide input of S0. Expected float, array, integer '
                'or one of: {0}'.format(S0s))
    # unit of latitude
    latunits = ['deg', 'rad']
    if lat_unit not in latunits:
        raise KeyError('Invalid latitude unit. Expected one of: {0}'
                       .format(latunits))
    # convert if needed
    if lat_unit == 'deg':
        lat = np.deg2rad(lat)

    # # translate dates to number of days of the year
    # length time-series
    T = (Tend - Tstart).days + 1
    # start day of the year
    nstart = (Tstart - datetime.datetime(Tstart.year, 1, 1).date()).days
    # days of the year
    n = np.arange(nstart, nstart + T) + 1

    # # constants - [deg] > [rad]
    # obliquity
    eps = np.deg2rad(eps)  # [rad]
    # longitude of perihelion
    lop = np.deg2rad(lop)  # [rad]

    # # calculations
    # polar angle of the Earth's surface
    theta = 2. * np.pi * ((n - n0) / 365.25)
    # Earth's declination
    d = eps * np.sin(theta)
    # mean distance between Earth and Sun / distance between Earth and Sun
    RORE = 1. + e * np.cos(theta - lop)
    # hour angle
    cr = np.tan(lat) * np.tan(d)
    try:
        if cr > 1.:
            h0 = np.pi
        elif cr < -1.:
            h0 = 0.
        else:
            h0 = np.arccos(- cr)
    except ValueError:
        h0 = np.zeros(len(n))
        h0[cr > 1] = np.pi
        h0[np.logical_and(cr >= -1, cr <= 1)] =\
            np.arccos(- cr[np.logical_and(cr >= -1, cr <= 1)])

    # solar constant
    if S0 == 'norm_lat':
        # normalisation S0
        Q = (1. / np.pi) * (RORE ** 2) * (h0 * np.sin(lat) * np.sin(d) +
                                          np.cos(lat) * np.cos(d) * np.sin(h0))
        # normalised solar constant
        S0 = 1. / Q.max()
    elif S0 == 'norm':
        # global light distribution
        Q = (1. / np.pi) * (RORE ** 2) * (np.pi * np.sin(d))
        # normalisation of S0
        S0 = 1. / Q.max()

    # daily insolation
    Id = (S0 / np.pi) * (RORE ** 2) * (h0 * np.sin(lat) * np.sin(d) +
                                       np.cos(lat) * np.cos(d) * np.sin(h0))

    # # write file
    if write:
        # directory
        filename = 'TS_PAR.txt'
        if path is None:
            filef = filename
        else:
            filef = os.path.join(path, filename)
        # time-series
        date = pd.date_range(Tstart, Tend, freq='D')
        TS = pd.DataFrame({'date': date,
                           'par': Id})
        # write to text-file
        TS.to_csv(filef, sep='\t', index=None)

    # # output
    if print_output:
        return Id


# =============================================================================
# # # # SST time-series
# =============================================================================


def TS_SST(lat, lon, Tstart, T,
           window=5,
           write=False, in_path=None, out_path=None,
           NOAAstart=1982, NOAAend=2019,
           print_output=True):
    """
    Daily average sea surface temperature (SST) due to the seasons, the
    location on Earth, and based on historic SST data (NOAA data set).

    Parameters
    ----------
    lat : numeric
        Latitude [DD coordinates].
    lon : numeric
        Longitude [DD coordinates].
    Tstart : datetime.date, datetime.datetime, string
        Start date of the SST time-series. If given as string, hold to the
        following formatting: str(YYYY-MM-DD) or str(DD-MM-YYYY).
    T : numeric
        Duration of SST time-series [yrs].
    window : integer, optional
        Number of days used for the rolling average [d].
    write : boolean, optional
        Write output file.
    in_path : string, optional
        Location of historic SST data sets.
    out_path, string, optional
        Location of output file (if [write] is enabled).
    NOAAstart : numeric, optional
        First full year of data of the NOAA data set (needed for statistics).
    NOAAend : numeric, optional
        Last full year of data of the NOAA data set (needed for statistics).
    print_output : boolean, optional
        Print the output to a parameter.

    Returns
    -------
    data : DataFrame
        SST time-series
    """
    # # check input
    # dates
    if isinstance(Tstart, (datetime.datetime, datetime.date)):
        pass
    elif isinstance(Tstart, str):
        Tstart = datetime.datetime.strptime(Tstart, '%Y-%m-%d').date()
    else:
        raise KeyError(
                'Wrong notation of start date. Type "TS_SST"? for help.')
    # window
    window = int(window)

    # # NOAA dataset (1981 - 2019)
    # define file
    [latN, lonN], [_, _] = support.data_coor(lat, lon)
    file = ('SST_timeseries_lat{0}lon{1}_Y1981_2019.txt'
            .format(latN, lonN))
    filef = os.path.join(in_path, file)
    # check its existence
    if os.path.isfile(filef):
        NOAA = pd.read_csv(filef, sep='\t')
    else:
        latN, lonN = support.SST_file_w(lat, lon, 1981, 2019,
                                        path=in_path, latlon=1)
        NOAA = pd.read_csv(filef, sep='\t')

    SST = NOAA[np.logical_and(NOAA['year'] >= NOAAstart,
                              NOAA['year'] < NOAAend)].reset_index(drop=True)

    # # SST statistics
    # annual statistics
    y = SST.groupby([SST.year])['sst'].agg(['mean'])
    y_ms = [y.mean()['mean'], y.std()['mean']]
    # monthly statistics w.r.t. annual statistics - monthly anomalies
    ym = SST.groupby([SST.year, SST.month])['sst'].agg(['mean'])
    ym.reset_index(inplace=True)
    ym.rename(columns={'mean': 'sst'}, inplace=True)
    ym = ym.merge(y.reset_index())
    ym['anom'] = ym['sst'] - ym['mean']
    ym.drop(['mean'], axis=1, inplace=True)
    ym.rename(columns={'sst': 'mean'}, inplace=True)
    m_ms = ym.groupby([ym.month])['anom'].agg(['mean', 'std'])
    # daily statistics w.r.t. monthly statistics - daily anomalies
    ymd = SST.merge(ym.drop(['anom'], axis=1))
    ymd['anom'] = ymd['sst'] - ymd['mean']
    d_ms = ymd.groupby([ymd.month, ymd.day])['anom'].agg(['mean', 'std'])

    # # SST predictions
    # end of time-series
    Tend = Tstart + dateutil.relativedelta.relativedelta(years=T, days=-1)
    # framework
    data = pd.DataFrame({'date': pd.date_range(Tstart, Tend, freq='D'),
                         'year': pd.date_range(Tstart, Tend, freq='D').year,
                         'month': pd.date_range(Tstart, Tend, freq='D').month,
                         'day': pd.date_range(Tstart, Tend, freq='D').day,
                         'dsst': 0.})
    # annual means
    SSTyear = np.random.normal(*y_ms, size=int(T))
    # monthly means
    SSTmonth = np.random.normal(m_ms['mean'].values,
                                m_ms['std'].values,
                                size=(int(T), 12)).flatten(order='C')

    # # from annual data to daily data
    data = pd.DataFrame({'year': np.arange(Tstart.year, Tend.year + 1),
                         'month': 1.,
                         'day': 1.,
                         'dsst': SSTyear})
    # set dates as index
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data.set_index('date', inplace=True)
    # # annual > monthly
    # reindex and forward fill - months
    dates = pd.date_range(Tstart, Tend, freq='MS')
    dates.name = 'date'
    data = data.reindex(dates, method='ffill')
    data['month'] = data.index.month
    # add monthly anomalies
    data['dsst'] += SSTmonth
    # # monthly > daily
    # reindex and forward fill - days
    dates = pd.date_range(Tstart, Tend, freq='D')
    dates.name = 'date'
    data = data.reindex(dates, method='ffill')
    data['day'] = data.index.day
    # add daily anomalies
    data = data.merge(d_ms.reset_index())
    data['dsst'] += np.random.normal(data['mean'], data['std'])
    # reset index to 'date'
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data.set_index('date', inplace=True)
    data.drop(['mean', 'std'], axis=1, inplace=True)
    data.sort_index(inplace=True)
    # rolling average
    data['sst'] = data['dsst'].rolling(window=window).mean()
    # remove dummy column
    data = data.drop(['dsst'], axis=1)

    # # write file
    if write:
        # directory
        filename = 'TS_SST.txt'
        if out_path is None:
            filef = filename
        else:
            filef = os.path.join(out_path, filename)
        # time-series
        TS = data['sst'].reset_index()
        # write to text-file
        TS.to_csv(filef, sep='\t', index=None)

    # # output
    if print_output:
        return data.drop(['year', 'month', 'day'], axis=1)


# =============================================================================
# # # # Hydrodynamic time-series
# =============================================================================


def TS_H(T, waveconditions, Tstart, dt, dt_storm=None,
         wind=False, windconditions=None,
         write_waves=True, MDWfile='wave', path_waves=None,
         print_stormcat=True, write_stormcat=False, path_stormcat=None):
    """
    Annual representative offshore wave conditions expressed in category number
    of the storm, where 0 represents the absence of storms; i.e. 'normal'
    conditions.

    Parameters
    ----------
    T : numeric, integer
        Duration of hydrodynamic time-series [yrs].
    waveconditions : array, list, DataFrame
        The wave conditions specified by a matrix with three columns: (1) the
        return period, R [yrs]; (2) the significant wave height, Hs [m];
        (3) the peak period, Tp [s]; (4) the direction, Hdir [deg]; and (5) the
        directional spreading, ms [deg]. The first row specifies the `normal'
        conditions and the return period shoudl be R = 1. From the second row
        onwards the storm conditions are specified. The probability of
        occurence is based on the given return periods. Thus, the format of the
        table of wave conditions is as follows:
            R  |  Hs  |  Tp  |  Hdir  |  ms
        ---------------------------------------
            1  |  Hs0 |  Tp0 |  Hdir0 |  ms0
            R1 |  Hs1 |  Tp1 |  Hdir1 |  ms1
           etc |  etc |  etc |  etc   |  etc
        ---------------------------------------
    Tstart : string, datetime.date, datetime.datetime
        Start date of the hydrodynamic time-series. If given as string, hold to
        the following formatting: str(YYYY-MM-DD) or str(DD-MM-YYYY).
    dt : numeric, optional
        Length of simulation time per set of wave conditions [min].
    write_waves : boolean, optional
        Write file with wave (and wind) conditions.
    wind : boolean, optional
        Include the effects of wind on the hydrodynamics [True] or not [False].
    windconditions : None, array, list, DataFrame, optional
        If [wind] is activated, a set of specifications on the wind conditions
        must be specified. The specifications include (1) the wind velocity,
        Uw [m s^-1]; and (2) the wind direction, Wdir [deg].
        The wind conditions must have the same length as the wave conditions,
        where the return periods as specified in the wave conditions are used
        for the specified wind conditions as well. Thus, the set of wind
        conditions must correspond with the set of wave conditions. Thus, the
        format of the table of wind conditions is as follows:
            Uw  |  Wdir
        -------------------
            Uw0 |  Wdir0
            Uw1 |  Wdir1
            etc |  etc
        -------------------
    MDWfile : None, string, optional
        Name of the wave-file without the extension; i.e. [MDWfile = {name}]
        for the wave file named as [ {name}.mdw ].
    path_waves : None, string, optional
        Path description to which the file with wave conditions must be
        written. When None, file is written in the same directory as this
        function is called.
    print_stormcat : boolean, optional
        Print the output to a parameter. If enabled, output is represented as
        the time-series of the storm categories. NOTE: the storm categories are
        not the same as the wave conditions associated with the different
        categories.
    write_stormcat : boolean, optional
        Write file with storm categories.
    path_stormcat : None, string, optional
        If write_stormcat is enabled, it is written in according this path
        description. When None, file is written in the same directory as this
        funciton is callled.

    Returns
    -------
    Time-series containing the wave conditions.
    """
    # # input check
    # wave conditions
    if not isinstance(waveconditions, (np.ndarray, list, pd.DataFrame)):
        raise KeyError(
                'Wrong formatting of wave conditions. See TS_H? for help.')
    # wind conditions
    if wind:
        if not isinstance(windconditions, (np.ndarray, list, pd.DataFrame)):
            raise KeyError(
                    'Wrong formatting of wind conditions. See TS_H? for help.')
    # start time
    if isinstance(Tstart, (datetime.datetime, datetime.date)):
        pass
    elif isinstance(Tstart, str):
        try:
            Tstart = datetime.datetime.strptime(Tstart, '%Y-%m-%d').date()
        except ValueError:
            Tstart = datetime.datetime.strptime(Tstart, '%d-%m-%Y').date()
    else:
        raise KeyError(
                'Wrong notation of start date. Type "TS_SST"? for help.')
    # storm time-step
    if dt_storm is None:
        dt_storm = dt

    # # wave and wind conditions formatted as matrix
    # wave conditions
    if isinstance(waveconditions, list):
        waveconditions = np.array(waveconditions)
    elif isinstance(waveconditions, pd.DataFrame):
        waveconditions = waveconditions.values
    # wind conditions
    if wind:
        if isinstance(windconditions, list):
            windconditions = np.array(windconditions)
        elif isinstance(windconditions, pd.DataFrame):
            windconditions = windconditions.values

    # # conditions on ascending order
    # wind conditions
    if wind:
        windconditions = windconditions[waveconditions[:, 0].argsort()]
    # wave conditions
    waveconditions = waveconditions[waveconditions[:, 0].argsort()]

    # # calculations
    # probabilities
    P = 1. / waveconditions[:, 0]
    # working array
    stormcat = np.zeros(int(T))
    # annual assessment
    for i in range(int(T)):
        pi = np.random.random()
        for j in range(len(P)):
            if pi < P[j]:
                stormcat[i] = j
            elif pi > P[0]:
                stormcat[i] = 0
    # add 'normal' conditions to start
    stormcat[0] = 0

    # # time-series
    # length time-series > two lines are needed per storm condition
    Tlen = int(T) + len(stormcat[stormcat > 0])
    # time-steps
    t = np.arange(0, dt * Tlen, dt)
    if dt_storm is not None:
        s = 0
        for i in range(int(T)):
            if stormcat[i] > 0:
                t[i + 1 + s::] += (dt_storm - dt)
                s += 1
    # time-series: wave conditions
    wave = np.zeros((int(Tlen), 4))
    s = 0
    for i in range(int(T)):
        if stormcat[i] > 0:
            wave[i + s, :] = waveconditions[int(stormcat[i]), 1:]
            s += 1
        wave[i + s, :] = waveconditions[0, 1:]
    # time-series: wind conditions
    wind_con = np.zeros((int(Tlen), 2))
    s = 0
    if wind:
        for i in range(len(P)):
            if stormcat[i] > 0:
                wind_con[i + s, :] = windconditions[stormcat[i], :]
                s += 1
            wind_con[i + s, :] = windconditions[0, :]

    # # write file(s)
    if write_waves:
        # write as DataFrame
        conditions = pd.DataFrame({'Time': np.append(
                                           int(Tlen), t),
                                   'Hs': np.append(
                                           '8', wave[:, 0]),
                                   'Tp': np.append(
                                           '', wave[:, 1]),
                                   'dir(deg)': np.append(
                                           '', wave[:, 2]),
                                   'ms(deg)': np.append(
                                           '', wave[:, 3]),
                                   'level': np.append(
                                           '', np.zeros(len(t))),
                                   'windv': np.append(
                                           '', wind_con[:, 0]),
                                   'winddir(deg)': np.append(
                                           '', wind_con[:, 1])})
        # file name and directory
        filename = 'wavecon.{0}'.format(MDWfile)
        if path_waves is None:
            filef = filename
        else:
            filef = os.path.join(path_waves, filename)
        # write to file
        conditions.to_csv(filef, index=None, sep=' ', mode='w')
        # warning
        print('\nWARNING: {0} not fully finished yet!'.format(filename))
        print('1. Add asterisk in front of "Time": "Time" -> "* Time".')
        print('2. Add line below heading with " BL01". Note the extra space.')

    if write_stormcat:
        # write as DataFrame
        stormcats = pd.DataFrame({'year': np.arange(int(Tstart.year),
                                                    int(Tstart.year + T)),
                                  'stormcat': stormcat})
        # file name and directory
        filename = 'TS_stormcat.txt'
        if path_stormcat is None:
            filef = filename
        else:
            filef = os.path.join(path_stormcat, filename)
        # write to file
        stormcats.to_csv(filef, index=None, sep='\t', mode='w')

    # # output
    if print_stormcat:
        return stormcat


def TS_ARAG(pCO2, S, T, pH=8.1, Tunit='deg.C'):
    """
    The aragonite saturation state as function of the atmospheric carbon
    dioxide pressure and other properties of the (sea) water. NOTE: The pH is
    not taken into account dynamically, but is enforced on the system based on
    a well-eduacted guess.

    Parameters
    ----------
    pCO2 : numeric or array
        Atmospheric carbon dioxide pressure [atm].
    S : numeric or array
        Salinity of water [ppt].
    T : numeric or array
        Temperature of water [K].
    pH : numeric or array, optional
        Acidity of water [-].
    Tunit : string, optional
        Unit of temperature used in SST time-series.
        Options: 'K', 'deg.C'

    Returns
    -------
    omega_a : numeric or array
        Aragonite saturation state [-].
    """
    # # input checks
    # temperature units check
    Tunits = ['K', 'deg.C']
    if Tunit not in Tunits:
        raise ValueError('Invalid temperature unit. Expected one of: {0}'
                         .format(Tunits))
    # convert to Kelvin (if necessary)
    if Tunit == 'deg.C':
        T += 273.15

    # # constants of the chemical reactions
    # K0*
    lnK0 = ((-60.2409 + 93.4517 * 100 / T + 23.3585 * np.log(T / 100)) +
            (.023517 - .023656 * T / 100 + .0047036 * (T / 100) ** 2) * S)
    K0 = np.exp(lnK0)
    # K1*
    lnK1 = ((2.83655 - 2307.1266 / T - 1.5529413 * np.log(T)) +
            (-.20760841 - 4.0484 / T) * S ** .5 +
            .1130822 * S -
            .00846934 * S ** 1.5)
    K1 = np.exp(lnK1)
    # K2*
    lnK2 = ((-9.226508 - 3351.6106 / T - .2005743 * np.log(T)) +
            (-.106901773 - 23.9722 / T) * S ** .5 +
            .1130822 * S -
            .00846934 * S ** 1.5)
    K2 = np.exp(lnK2)
    # Ka*
    logKa = ((-171.945 - .077995 * T + 2909.298 / T + 71.595 * np.log10(T)) +
             (-.068393 + .0017276 * T + 88.135 / T) * S ** .5 -
             .10018 * S +
             .0059415 * S ** 1.5)
    Ka = 10 ** (logKa)

    # # H+ concentration (pH)
    h = 10 ** (-pH)

    # # chemical reactions
    # dissolved CO3
    co3 = K0 * K1 * K2 * pCO2 / (h ** 2)
    # dissolved Ca
    ca = .01028 * S / 35
    # aragonite saturation state
    omega = (ca * co3) / Ka

    # # output
    if Tunit == 'deg.C':
        T -= 273.15

    return omega


# =============================================================================
# =============================================================================
# # # # # Time-series
# =============================================================================
# =============================================================================

if __name__ = '__main__':
    # =========================================================================
    # # # # figure formatting
    # =========================================================================
    # # save files/figures and show figures
    save = 0
    fig = 1

    figname = 'TimeSeries'
    figpath = os.path.join('..', 'Figures', 'Python_figures')

    # # png-settings
    figext = '.png'
    fs = 8
    lw = .5
    figqual = 600  # dpi

    # # figure dimensions
    figheight = 12.  # in
    figwidth = 8.  # in

    # # LaTeX format
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # # basic color definitions
    bish = (0., 0., 128. / 255.)
    rish = (128. / 255., 0., 0.)
    gish = (0., 128. / 255., 0.)
    # matrix-format
    colors = np.array([bish, rish, gish])

    # =========================================================================
    # # # # data
    # =========================================================================
    # # constants
    pCO2 = 400e-6
    S = 35.
    pH = 8.

    # # directory
    model_folder = 'MinFiles'
    workdir = os.path.join('p:\\11202744-008-vegetation-modelling', 'students',
                           'GijsHendrickx', 'models', model_folder)
    tspath = os.path.join(workdir, 'timeseries')
    wavepath = os.path.join(workdir, 'wave')

    # check existence and create if necessary
    if not os.path.exists(tspath):
        os.mkdir(tspath)
        print('New folder created : {0}'.format(tspath))
    if not os.path.exists(wavepath):
        os.mkdir(wavepath)
        print('New folder created : {0}'.format(wavepath))

    # # time-settings
    Tstart = datetime.date(2000, 1, 1)
    Tend = datetime.date(2099, 12, 31)
    T = Tend.year - Tstart.year + 1.
    # historic time-series > thermal limits [yrs] (for TS_SST and TS_PAR)
    n = 60
    T0 = Tstart - dateutil.relativedelta.relativedelta(years=n)

    # # spatial settings
    lat = -15.
    lon = 145.5

    # # wave conditions
    waves = pd.DataFrame({'R': [1., 10., 50., 100.],
                          'Hs': [1.2, 2., 3.5, 4.],
                          'Tp': [4., 4., 5., 6.],
                          'dir': [180., 180., 180., 180.],
                          'ms': [2., 2., 2., 2.]})

    # # Time-series
    # PAR
    Id = TS_PAR(
            T0, Tend, lat, write=save, path=tspath, print_output=fig)
    # hydrodynamics
    stormcat = TS_H(
            T, waves, Tstart, dt=43200, dt_storm=86400,
            write_waves=save, path_waves=wavepath,
            print_stormcat=fig, write_stormcat=save, path_stormcat=tspath)
    # SST
    sst = TS_SST(
            lat, lon, T0, T + n, window=5, write=save, in_path='SST_data',
            out_path=tspath, print_output=fig)
    # aragonite
    om = TS_ARAG(pCO2, S, sst['sst'].values, pH)
    omega = pd.DataFrame({'date': pd.date_range(T0, Tend, freq='D'),
                          'omega': om})
    omega.set_index('date', inplace=True)

    # =========================================================================
    # # # # figure
    # =========================================================================
    # # data
    t = pd.date_range(T0, Tend, freq='D')
    y = pd.date_range(T0, Tend, freq='YS')

    # # figure
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False,
                           figsize=(figwidth, figheight))
    plt.subplots_adjust(hspace=0)
    # zero-line
    # plot data
    ax[0].plot(
            t[t.date >= Tstart], Id[t.date >= Tstart],
            color=bish, alpha=.8,
            linewidth=lw)
    ax[1].scatter(
            y[y.date >= Tstart], stormcat,
            color=bish, alpha=.8,
            s=10)
    ax[2].plot(
            t[t.date < Tstart], sst[t.date < Tstart],
            color='gray', alpha=.5,
            linewidth=lw)
    ax[2].plot(
            t[t.date >= Tstart], sst[t.date >= Tstart],
            color=rish, alpha=.8,
            linewidth=lw)
    ax[3].plot(
            t[t.date >= Tstart], omega[t.date >= Tstart],
            color=gish, alpha=.8,
            linewidth=lw)
    # axes labels
    ax[0].set_ylabel(r'$I_{{0}}$ [$\mu E m^{{-2}} s^{{-1}}$]')
    ax[1].set_ylabel('storm category')
    ax[2].set_ylabel(r'$SST$ [${{}}^{{\circ}}$C]')
    ax[3].set_ylabel(r'$\Omega_{{a}}$ [$-$]')
    ax[-1].set_xlabel(r'date')
    fig.align_labels()
    # axes ticks
    ax[1].set_yticks(np.arange(0, 4))
    # plot limits
    ax[0].set_xlim([t.min(), t.max()])
    ax[1].set_ylim([-.1, 3.1])
    # explaining lines / texts / etc.
    # legend / title

    # # save figure
    if save:
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        fig.savefig(figfull, dpi=figqual,
                    bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))
