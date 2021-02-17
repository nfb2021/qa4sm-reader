# -*- coding: utf-8 -*-
"""
Contains helper functions for plotting qa4sm results.
"""
from qa4sm_reader import globals
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from cartopy import config as cconfig
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import warnings
from pygeogrids.grids import BasicGrid, genreg_grid
cconfig['data_dir'] = os.path.join(os.path.dirname(__file__), 'cartopy')

def _float_gcd(a, b, atol=1e-08):
    "Greatest common divisor (=groesster gemeinsamer teiler)"
    while abs(b) > atol:
        a, b = b, a % b
    return a

def _get_grid(a):
    "Find the stepsize of the grid behind a and return the parameters for that grid axis."
    a = np.unique(a)  # get unique values and sort
    das = np.unique(np.diff(a))  # get unique stepsizes and sort
    da = das[0]  # get smallest stepsize
    for d in das[1:]:  # make sure, all stepsizes are multiple of da
        da = _float_gcd(d, da)
    a_min = a[0]
    a_max = a[-1]
    len_a = int((a_max - a_min) / da + 1)
    return a_min, a_max, da, len_a

def _get_grid_for_irregulars(a, grid_stepsize):
    "Find the stepsize of the grid behind a for datasets with predeifned grid stepsize, and return the parameters for that grid axis."
    a = np.unique(a)
    a_min = a[0]
    a_max = a[-1]
    da = grid_stepsize
    len_a = int((a_max - a_min) / da + 1)
    return a_min, a_max, da, len_a

def _value2index(a, a_min, da):
    "Return the indexes corresponding to a. a and the returned index is a numpy array."
    return ((a - a_min) / da).astype('int')

def oversample(lon, lat, data, extent, dx, dy):

    other = BasicGrid(lon, lat)
    reg_grid = genreg_grid(dx, dy, minlat=extent[2], maxlat=extent[3],
                           minlon=extent[0], maxlon=extent[1])
    max_dist = dx * 111 * 1000 # a mean distance for one degree it's around 111 km
    lut = reg_grid.calc_lut(other, max_dist=max_dist)
    img = np.ma.masked_where(lut == -1, data[lut])
    img[np.isnan(img)] = np.ma.masked

    return img.reshape(-1, reg_grid.shape[1]), reg_grid

def geotraj_to_geo2d(df, var, index=globals.index_names, grid_stepsize=None):
    """
    Converts geotraj (list of lat, lon, value) to a regular grid over lon, lat.
    The values in df needs to be sampled from a regular grid, the order does not matter.
    When used with plt.imshow(), specify data_extent to make sure, 
    the pixels are exactly where they are expected.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'lat', 'lon' and 'var' Series.
    var : str
        variable to be converted.
    index : tuple, optional
        Tuple containing the names of lattitude and longitude index. Usually ('lat','lon')
        The default is globals.index_names
    grid_stepsize : None or float, optional
        angular grid stepsize to prepare a regular grid for plotting

    Returns
    -------
    zz : numpy.ndarray
        array holding the gridded values. When using plt.imshow, specify origin='lower'.
        [0,0] : llc (lower left corner)
        first coordinate is longitude.
    data_extent : tuple
        (x_min, x_max, y_min, y_max) in Data coordinates.
    origin : string
        'upper' or 'lower' - define how the plot should be oriented, for irregular grids it should return 'upper'
    """
    xx = df.index.get_level_values(index[1])  # lon
    yy = df.index.get_level_values(index[0])   # lat
    data = df[var]

    if grid_stepsize not in ['nan', None]:
        x_min, x_max, dx, len_x = _get_grid_for_irregulars(xx, grid_stepsize)
        y_min, y_max, dy, len_y = _get_grid_for_irregulars(yy, grid_stepsize)
        data_extent = (x_min - dx/2, x_max + dx/2, y_min - dy/2, y_max + dy/2)
        zz, grid = oversample(xx, yy, data.values, data_extent, dx, dy)
        origin = 'upper'
    else:
        x_min, x_max, dx, len_x = _get_grid(xx)
        y_min, y_max, dy, len_y = _get_grid(yy)
        ii = _value2index(yy, y_min, dy)
        jj = _value2index(xx, x_min, dx)
        zz = np.full((len_y, len_x), np.nan, dtype=np.float64)
        zz[ii, jj] = data
        data_extent = (x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2)
        origin = 'lower'

    return zz, data_extent, origin

def get_value_range(ds, metric=None, force_quantile=False, quantiles=[0.025, 0.975]):
    """
    Get the value range (v_min, v_max) from globals._metric_value_ranges
    If the range is (None, None), a symmetric range around 0 is created,
    showing at least the symmetric <quantile> quantile of the values.
    if force_quantile is True, the quantile range is used.

    Parameters
    ----------
    ds : pd.DataFrame or pd.Series
        Series holding the values
    metric : str , optional (default: None)
        name of the metric (e.g. 'R'). None equals to force_quantile=True.
    force_quantile : bool, optional
        always use quantile, regardless of globals.
        The default is False.
    quantiles : list, optional
        quantile of data to include in the range.
        The default is [0.025,0.975]

    Returns
    -------
    v_min : float
        lower value range of plot.
    v_max : float
        upper value range of plot.
    """
    if metric == None:
        force_quantile = True

    if not force_quantile:  # try to get range from globals
        try:
            v_min = globals._metric_value_ranges[metric][0]
            v_max = globals._metric_value_ranges[metric][1]
            if (v_min is None and v_max is None):  # get quantile range and make symmetric around 0.
                v_min, v_max = get_quantiles(ds, quantiles)
                v_max = max(abs(v_min), abs(v_max))  # make sure the range is symmetric around 0
                v_min = -v_max
            elif v_min is None:
                v_min = get_quantiles(ds, quantiles)[0]
            elif v_max is None:
                v_max = get_quantiles(ds, quantiles)[1]
            else:  # v_min and v_max are both determinded in globals
                pass
        except KeyError:  # metric not known, fall back to quantile
            force_quantile = True
            warnings.warn('The metric \'{}\' is not known. \n'.format(metric) + \
                          'Could not get value range from globals._metric_value_ranges\n' + \
                          'Computing quantile range \'{}\' instead.\n'.format(str(quantiles)) +
                          'Known metrics are: \'' + \
                          '\', \''.join([metric for metric in globals._metric_value_ranges]) + '\'')

    if force_quantile:  # get quantile range
        v_min, v_max = get_quantiles(ds, quantiles)

    return v_min, v_max

def get_quantiles(ds, quantiles):
    """
    Gets lower and upper quantiles from pandas.Series or pandas.DataFrame

    Parameters
    ----------
    ds : (pandas.Series | pandas.DataFrame)
        Input values.
    quantiles : list
        quantile of values to include in the range

    Returns
    -------
    v_min : float
        lower quantile.
    v_max : float
        upper quantile.

    """
    q = ds.quantile(quantiles)
    if isinstance(ds, pd.Series):
        return q.iloc[0], q.iloc[1]
    elif isinstance(ds, pd.DataFrame):
        return min(q.iloc[0]), max(q.iloc[1])
    else:
        raise TypeError("Inappropriate argument type. 'ds' must be pandas.Series or pandas.DataFrame.")

def get_plot_extent(df, grid_stepsize=None, grid=False):
    """
    Gets the plot_extent from the values. Uses range of values and
    adds a padding fraction as specified in globals.map_pad

    Parameters
    ----------
    grid : bool
        whether the values in df is on a equally spaced grid (for use in mapplot)
    df : pandas.DataFrame
        Plot values.
    
    Returns
    -------
    extent : tuple | list
        (x_min, x_max, y_min, y_max) in Data coordinates.
    
    """
    lat, lon = globals.index_names
    if grid and grid_stepsize in ['nan', None]:
        x_min, x_max, dx, len_x = _get_grid(df.index.get_level_values(lon))
        y_min, y_max, dy, len_y = _get_grid(df.index.get_level_values(lat))
        extent = [x_min-dx/2., x_max+dx/2., y_min-dx/2., y_max+dx/2.]
    elif grid and grid_stepsize:
        x_min, x_max, dx, len_x = _get_grid_for_irregulars(df.index.get_level_values(lon), grid_stepsize)
        y_min, y_max, dy, len_y = _get_grid_for_irregulars(df.index.get_level_values(lat), grid_stepsize)
        extent = [x_min - dx / 2., x_max + dx / 2., y_min - dx / 2., y_max + dx / 2.]
    else:
        extent = [df.index.get_level_values(lon).min(), df.index.get_level_values(lon).max(),
                  df.index.get_level_values(lat).min(), df.index.get_level_values(lat).max()]
    dx = extent[1] - extent[0]
    dy = extent[3] - extent[2]
    # set map-padding around values to be globals.map_pad percent of the smaller dimension
    padding = min(dx, dy) * globals.map_pad / (1 + globals.map_pad)
    extent[0] -= padding
    extent[1] += padding
    extent[2] -= padding
    extent[3] += padding
    if extent[0] < -180:
        extent[0] = -180
    if extent[1] > 180:
        extent[1] = 180
    if extent[2] < -90:
        extent[2] = -90
    if extent[3] > 90:
        extent[3] = 90
    return extent

def init_plot(figsize, dpi, add_cbar=None, projection=None):
    if not projection:
        projection=globals.crs
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if add_cbar:
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[19, 1])
        ax = fig.add_subplot(gs[0], projection=projection)
        cax = fig.add_subplot(gs[1])
    else:
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax = fig.add_subplot(gs[0], projection=projection)
        cax = None
    return fig, ax, cax

def get_extend_cbar(metric):
    """
    Find out whether the colorbar should extend, based on globals._metric_value_ranges[metric]

    Parameters
    ----------
    metric : str
        metric used in plot

    Returns
    -------
    str
        one of ['neither', 'min', 'max', 'both'].
    """
    vrange = globals._metric_value_ranges[metric]
    if vrange[0] is None:
        if vrange[1] is None:
            return 'both'
        else:
            return 'min'
    else:
        if vrange[1] is None:
            return 'max'
        else:
            return 'neither'

def style_map(ax, plot_extent, add_grid=True, map_resolution=globals.naturalearth_resolution,
              add_topo=False, add_coastline=True,
              add_land=True, add_borders=True, add_us_states=False):
    ax.set_extent(plot_extent, crs=globals.data_crs)
    ax.outline_patch.set_linewidth(0.4)
    if add_grid:
        # add gridlines. Bcs a bug in cartopy, draw girdlines first and then grid labels.
        # https://github.com/SciTools/cartopy/issues/1342
        try:
            grid_interval = max((plot_extent[1] - plot_extent[0]),
                                (plot_extent[3] - plot_extent[2])) / 5  # create apprx. 5 gridlines in the bigger dimension
            if grid_interval <= min(globals.grid_intervals):
                raise RuntimeError
            grid_interval = min(globals.grid_intervals, key=lambda x: abs(
                x - grid_interval))  # select the grid spacing from the list which fits best
            gl = ax.gridlines(crs=globals.data_crs, draw_labels=False,
                              linewidth=0.5, color='grey', linestyle='--',
                              zorder=3)  # draw only gridlines.
            # todo this can slow the plotting down!!
            xticks = np.arange(-180, 180.001, grid_interval)
            yticks = np.arange(-90, 90.001, grid_interval)
            gl.xlocator = mticker.FixedLocator(xticks)
            gl.ylocator = mticker.FixedLocator(yticks)
        except RuntimeError:
            pass
        else:
            try:  # drawing labels fails for most projections
                gltext = ax.gridlines(crs=globals.data_crs, draw_labels=True,
                                      linewidth=0.5, color='grey', alpha=0., linestyle='-',
                                      zorder=4)  # draw only grid labels.
                xticks = xticks[(xticks >= plot_extent[0]) & (xticks <= plot_extent[1])]
                yticks = yticks[(yticks >= plot_extent[2]) & (yticks <= plot_extent[3])]
                gltext.xformatter = LONGITUDE_FORMATTER
                gltext.yformatter = LATITUDE_FORMATTER
                gltext.top_labels = False
                gltext.left_labels = False
                gltext.xlocator = mticker.FixedLocator(xticks)
                gltext.ylocator = mticker.FixedLocator(yticks)
            except RuntimeError as e:
                print("No tick labels plotted.\n" + str(e))
    if add_topo:
        ax.stock_img()
    if add_coastline:
        coastline = cfeature.NaturalEarthFeature('physical', 'coastline',
                                                 map_resolution,
                                                 edgecolor='black', facecolor='none')
        ax.add_feature(coastline, linewidth=0.4, zorder=3)
    if add_land:
        land = cfeature.NaturalEarthFeature('physical', 'land',
                                            map_resolution,
                                            edgecolor='none', facecolor='white')
        ax.add_feature(land, zorder=1)
    if add_borders:
        borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries',
                                               map_resolution,
                                               edgecolor='black', facecolor='none')
        ax.add_feature(borders, linewidth=0.2, zorder=3)
    if add_us_states:
        ax.add_feature(cfeature.STATES, linewidth=0.1, zorder=3)

    return ax

def make_watermark(fig, placement=globals.watermark_pos, for_map=False, offset=0.02):
    """
    Adds a watermark to fig and adjusts the current axis to make sure there
    is enough padding around the watermarks.
    Padding can be adjusted in globals.watermark_pad.
    Fontsize can be adjusted in globals.watermark_fontsize.
    plt.tight_layout needs to be called prior to make_watermark,
    because tight_layout does not take into account annotations.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    placement : str
        'top' : places watermark in top right corner
        'bottom' : places watermark in bottom left corner
    """
    # ax = fig.gca()
    # pos1 = ax.get_position() #fraction of figure
    fontsize = globals.watermark_fontsize
    pad = globals.watermark_pad
    height = fig.get_size_inches()[1]
    offset = offset + (((fontsize + pad) / globals.matplotlib_ppi) / height) * 2.2
    if placement == 'top':
        plt.annotate(globals.watermark, xy=[0.5, 1], xytext=[-pad, -pad],
                     fontsize=fontsize, color='grey',
                     horizontalalignment='center', verticalalignment='top',
                     xycoords='figure fraction', textcoords='offset points')
        top = fig.subplotpars.top
        fig.subplots_adjust(top=top - offset)
    elif placement == 'bottom':
        plt.annotate(globals.watermark, xy=[0.5, 0], xytext=[pad, pad],
                     fontsize=fontsize, color='grey',
                     horizontalalignment='center', verticalalignment='bottom',
                     xycoords='figure fraction', textcoords='offset points')
        bottom = fig.subplotpars.bottom
        if not for_map:
            fig.subplots_adjust(bottom=bottom + offset)  # defaults to rc when none!
    else:
        raise NotImplementedError