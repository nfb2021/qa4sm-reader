# -*- coding: utf-8 -*-
"""
Contains helper functions for plotting qa4sm results.
"""
from qa4sm_reader import globals
import numpy as np
import pandas as pd
import os.path
import copy

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, PathPatch
from matplotlib.lines import Line2D
from cartopy import config as cconfig
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pygeogrids.grids import BasicGrid, genreg_grid
from shapely.geometry import Polygon, Point

import warnings

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

def _format_floats(x):
    """Format floats in the statistsics table"""
    if isinstance(x, float):
        if abs(x) < 0.000001:
            return "~ 0"
        elif 0.1 < abs(x) < 1e3:
            return np.format_float_positional(x, precision=2)
        else:
            return np.format_float_scientific(x, precision=2)
    else:
        return x

def oversample(lon, lat, data, extent, dx, dy):

    other = BasicGrid(lon, lat)
    reg_grid = genreg_grid(dx, dy, minlat=extent[2], maxlat=extent[3],
                           minlon=extent[0], maxlon=extent[1])
    max_dist = dx * 111 * 1000 # a mean distance for one degree it's around 111 km
    lut = reg_grid.calc_lut(other, max_dist=max_dist)
    img = np.ma.masked_where(lut == -1, data[lut])
    img[np.isnan(img)] = np.ma.masked

    return img.reshape(-1, reg_grid.shape[1]), reg_grid

def geotraj_to_geo2d(df, index=globals.index_names, grid_stepsize=None):
    """
    Converts geotraj (list of lat, lon, value) to a regular grid over lon, lat.
    The values in df needs to be sampled from a regular grid, the order does not matter.
    When used with plt.imshow(), specify data_extent to make sure, 
    the pixels are exactly where they are expected.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'lat', 'lon' and 'var' Series.
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

    if grid_stepsize not in ['nan', None]:
        x_min, x_max, dx, len_x = _get_grid_for_irregulars(xx, grid_stepsize)
        y_min, y_max, dy, len_y = _get_grid_for_irregulars(yy, grid_stepsize)
        data_extent = (x_min - dx/2, x_max + dx/2, y_min - dy/2, y_max + dy/2)
        zz, grid = oversample(xx, yy, df.values, data_extent, dx, dy)
        origin = 'upper'
    else:
        x_min, x_max, dx, len_x = _get_grid(xx)
        y_min, y_max, dy, len_y = _get_grid(yy)
        ii = _value2index(yy, y_min, dy)
        jj = _value2index(xx, x_min, dx)
        zz = np.full((len_y, len_x), np.nan, dtype=np.float64)
        zz[ii, jj] = df
        data_extent = (x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2)
        origin = 'lower'

    return zz, data_extent, origin

def get_value_range(ds, metric=None, force_quantile=False, quantiles=[0.025, 0.975], diff_map=False):
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
    diff_map : bool, default is False
        Whether the colorbar is for a difference plot

    Returns
    -------
    v_min : float
        lower value range of plot.
    v_max : float
        upper value range of plot.
    """
    if metric == None:
        force_quantile = True

    ranges = globals._metric_value_ranges
    if not force_quantile:  # try to get range from globals
        try:
            v_min = ranges[metric][0]
            v_max = ranges[metric][1]
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
                          '\', \''.join([metric for metric in ranges]) + '\'')

    if force_quantile:  # get quantile range
        v_min, v_max = get_quantiles(ds, quantiles)
        # adjust range based on the difference values in the map
        if diff_map:
            extreme = max([abs(v) for v in get_quantiles(ds, quantiles)])
            v_min, v_max = -extreme, extreme

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
    lat, lon, gpi = globals.index_names
    if grid and grid_stepsize in ['nan', None]:
        # todo: problem if only single lon/lat point is present?
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

def style_map(
        ax, plot_extent, add_grid=True,
        map_resolution=globals.naturalearth_resolution,
        add_topo=False, add_coastline=True,
        add_land=True, add_borders=True, add_us_states=False,
        grid_intervals=globals.grid_intervals,
):
    ax.set_extent(plot_extent, crs=globals.data_crs)
    ax.spines["geo"].set_linewidth(0.4)
    if add_grid:
        # add gridlines. Bcs a bug in cartopy, draw girdlines first and then grid labels.
        # https://github.com/SciTools/cartopy/issues/1342
        try:
            grid_interval = max((plot_extent[1] - plot_extent[0]),
                                (plot_extent[3] - plot_extent[2])) / 5  # create apprx. 5 gridlines in the bigger dimension
            if grid_interval <= min(grid_intervals):
                raise RuntimeError
            grid_interval = min(grid_intervals, key=lambda x: abs(
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
                gltext.right_labels = False
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

def make_watermark(fig, placement=globals.watermark_pos, for_map=False, offset=0.03):
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
            fig.subplots_adjust(bottom=bottom + offset)
    else:
        raise NotImplementedError

def _make_cbar(fig, im, cax, ref_short:str, metric:str, label=None, diff_map=False):
    """
    Make colorbar to use in plots

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure of plot
    im: AxesImage
        from method Axes.imshow()
    cax: axes.SubplotBase
        from fig.add_subplot
    ref_short: str
        name of ref dataset
    metric: str
        name of metric
    label: str
        label to describe the colorbar
    diff_map : bool, default is False
        Whether the colorbar is for a difference plot
    """
    if label is None:
        try:
            label = globals._metric_name[metric] + \
                    globals._metric_description[metric].format(
                        globals._metric_units[ref_short])
        except KeyError as e:
            raise Exception('The metric \'{}\' or reference \'{}\' is not known.\n'.format(metric, ref_short) + str(e))

    extend = get_extend_cbar(metric)
    if diff_map:
        extend = "both"
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
    cbar.set_label(label, weight='normal')
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor('black')
    cbar.ax.tick_params(width=0.4)

    return fig, im, cax

def _CI_difference(fig, ax, ci):
    """
    Insert the median value of the upper and lower CI difference

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure with CIs
    ci: list
        list of upper and lower ci dataframes
    """
    lower_pos = []
    for ax in fig.axes:
        n = 0
        # iterating through axes artists:
        for c in ax.get_children():
            # searching for PathPatches
            if isinstance(c, PathPatch):
                # different width whether it's the metric or the CIs
                if n in np.arange(0, 100, 3):
                    # getting current width of box:
                    p = c.get_path()
                    verts = p.vertices
                    verts_sub = verts[:-1]
                    xmin = np.min(verts_sub[:, 0])
                    lower_pos.append(xmin)
                n += 1
    for ci_df, xmin in zip(ci, lower_pos):
        diff = ci_df["upper"] - ci_df["lower"]
        ci_range = float(diff.mean())
        ypos = float(ci_df["lower"].min())
        ax.annotate(
            "Mean CI\nRange:\n {:.2g}".format(ci_range),
            xy = (xmin - 0.2, ypos),
            horizontalalignment="center"
        )

def _add_dummies(df:pd.DataFrame, to_add:int) -> list:
    """
    Add empty columns in dataframe to avoid error in matplotlib when not all boxplot groups have the same
    number of values
    """
    for n, col in enumerate(np.arange(to_add)):
        # add columns while avoiding name clashes
        df[str(n)] = np.nan

    return df

def patch_styling(
        box_dict,
        facecolor
) -> None:
    """Define style of the boxplots"""
    for n, (patch, median) in enumerate(zip(box_dict["boxes"], box_dict["medians"])):
        patch.set(color="grey", facecolor=facecolor, linewidth=1.6, alpha=0.7)
        median.set(color="grey", linewidth=1.6)
    for (whis, caps) in zip(box_dict["whiskers"], box_dict["caps"]):
        whis.set(color="grey", linewidth=1.6)
        caps.set(color="grey", linewidth=1.6)

def boxplot(
        df,
        ci=None,
        label=None,
        figsize=None,
        dpi=100,
        spacing=0.35,
        **kwargs
):
    """
    Create a boxplot_basic from the variables in df.
    The box shows the quartiles of the dataset while the whiskers extend
    to show the rest of the distribution, except for points that are
    determined to be “outliers” using a method that is a function of
    the inter-quartile range.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'lat', 'lon' and (multiple) 'var' Series.
    ci : list
        list of Dataframes containing "upper" and "lower" CIs
    label : str, optional
        Label of the y axis, describing the metric. The default is None.
    figsize : tuple, optional
        Figure size in inches. The default is globals.map_figsize.
    dpi : int, optional
        Resolution for raster graphic output. The default is globals.dpi.
    spacing : float, optional.
        Space between the central boxplot and the CIs. Default is 0.3

    Returns
    -------
    fig : matplotlib.figure.Figure
        the boxplot
    ax : matplotlib.axes.Axes
    """
    values = df.copy()
    # make plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    center_pos = np.arange(len(values.columns))*2
    # styling
    ticklabels = values.columns
    if kwargs is None:
        kwargs = {}
    kwargs.update(patch_artist=True, return_type="dict")
    # changes necessary to have confidence intervals in the plot
    if ci:
        upper, lower = [], []
        for n, intervals in enumerate(ci):
            lower.append(intervals["lower"])
            upper.append(intervals["upper"])
        lower = _add_dummies(
            pd.concat(lower, ignore_index=True, axis=1),
            len(center_pos)-len(ci),
        )
        upper = _add_dummies(
            pd.concat(upper, ignore_index=True, axis=1),
            len(center_pos)-len(ci),
        )
        low = lower.boxplot(
            positions=center_pos - spacing,
            showfliers=False,
            widths=0.15,
            **kwargs
        )
        up = upper.boxplot(
            positions=center_pos + spacing,
            showfliers=False,
            widths=0.15,
            **kwargs
        )
        patch_styling(low, 'skyblue')
        patch_styling(up, 'tomato')
    # create plot
    cen = values.boxplot(
        positions=center_pos,
        showfliers=False,
        widths=0.3,
        **kwargs
    )
    patch_styling(cen, 'white')
    plt.xticks(center_pos, ticklabels)
    if ci:
        low_ci = Patch(color='skyblue', alpha=0.7, label='Lower CI')
        up_ci = Patch(color='tomato',  alpha=0.7, label='Upper CI')
        #_CI_difference(fig, ax, ci)
        plt.legend(
            handles=[low_ci, up_ci],
            fontsize=8,
            loc="best"
        )
    # provide y label
    if label is not None:
        plt.ylabel(label, weight='normal')
    plt.grid(axis='x')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, ax

def mapplot(
        df, metric,
        ref_short,
        ref_grid_stepsize=None,
        plot_extent=None,
        colormap=None,
        projection=None,
        add_cbar=True,
        label=None,
        figsize=globals.map_figsize,
        dpi=globals.dpi,
        diff_map=False,
        **style_kwargs
):
        """
        Create an overview map from df using values as color. Plots a scatterplot for ISMN and an image plot for other
        input values.

        Parameters
        ----------
        df : pandas.Series
            values to be plotted. Generally from metric_df[Var]
        metric : str
            name of the metric for the plot
        ref_short : str
                short_name of the reference dataset (read from netCDF file)
        ref_grid_stepsize : float or None, optional (None by default)
                angular grid stepsize, needed only when ref_is_angular == False,
        plot_extent : tuple
                (x_min, x_max, y_min, y_max) in Data coordinates. The default is None.
        colormap :  Colormap, optional
                colormap to be used.
                If None, defaults to globals._colormaps.
        projection :  cartopy.crs, optional
                Projection to be used. If none, defaults to globals.map_projection.
                The default is None.
        add_cbar : bool, optional
                Add a colorbar. The default is True.
        label : str, optional
            Label of the y axis, describing the metric. If None, a label is autogenerated from metadata.
            The default is None.
        figsize : tuple, optional
            Figure size in inches. The default is globals.map_figsize.
        dpi : int, optional
            Resolution for raster graphic output. The default is globals.dpi.
        diff_map : bool, default is False
            if True, a difference colormap is created
        **style_kwargs :
            Keyword arguments for plotter.style_map().

        Returns
        -------
        fig : matplotlib.figure.Figure
            the boxplot
        ax : matplotlib.axes.Axes
        """
        if not colormap:
            cmap = globals._colormaps[metric]
        else:
            cmap = colormap
        v_min, v_max = get_value_range(df, metric)
        # everything changes if the plot is a difference map
        if diff_map:
            v_min, v_max = get_value_range(df, metric=None, diff_map=True)
            cmap = globals._diff_colormaps[metric]

        # mask values outside range (e.g. for negative STDerr from TCA)
        if metric in globals._metric_mask_range.keys():
            mask_under, mask_over = globals._metric_mask_range[metric]  # get values from scratch to disregard quantiles
            cmap = copy.copy(cmap)
            if mask_under is not None:
                v_min = mask_under
                cmap.set_under("red")
            if mask_over is not None:
                v_max = mask_over
                cmap.set_over("red")

        # initialize plot
        fig, ax, cax = init_plot(figsize, dpi, add_cbar, projection)

        # scatter point or mapplot
        if ref_short in globals.scattered_datasets:  # scatter
            if not plot_extent:
                plot_extent = get_plot_extent(df)

            markersize = globals.markersize ** 2
            lat, lon, gpi = globals.index_names
            im = ax.scatter(df.index.get_level_values(lon),
                            df.index.get_level_values(lat),
                            c=df, cmap=cmap, s=markersize,
                            vmin=v_min, vmax=v_max,
                            edgecolors='black', linewidths=0.1,
                            zorder=2, transform=globals.data_crs)
        else:  # mapplot
            if not plot_extent:
                plot_extent = get_plot_extent(df, grid_stepsize=ref_grid_stepsize, grid=True)
            if isinstance(ref_grid_stepsize, np.ndarray):
                ref_grid_stepsize = ref_grid_stepsize[0]
            zz, zz_extent, origin = geotraj_to_geo2d(df, grid_stepsize=ref_grid_stepsize)  # prep values
            im = ax.imshow(zz, cmap=cmap, vmin=v_min,
                           vmax=v_max, interpolation='nearest',
                           origin=origin, extent=zz_extent,
                           transform=globals.data_crs, zorder=2)

        if add_cbar:  # colorbar
            _make_cbar(fig, im, cax, ref_short, metric, label=label, diff_map=diff_map)
        style_map(ax, plot_extent, **style_kwargs)
        fig.canvas.draw()  # very slow. necessary bcs of a bug in cartopy: https://github.com/SciTools/cartopy/issues/1207

        return fig, ax

def plot_spatial_extent(
        polys:dict,
        ref_points:bool=None,
        overlapping:bool=False,
        intersection_extent:tuple=None,
        reg_grid=False,
        grid_stepsize=None,
        **kwargs,
):
    """
    Plots the given Polygons and optionally the reference points on a map.

    Parameters
    ----------
    polys : dict
        dictionary with shape {name: shapely.geometry.Polygon}
    ref_points : 2D array
        array of lon, lat for the reference points positions
    overlapping : bool, dafault is False.
        Whether the polygons have an overlap
    intersection_extent : tuple | None
        if given, corresponds to the extent of the intersection. Shape (minlon, maxlon, minlat, maxlat)
    reg_grid : bool, default is False,
        plotting oprion for regular grids (satellites)
    """
    fig, ax, cax = init_plot(figsize=globals.map_figsize, dpi=globals.dpi)
    legend_elements = []
    # plot polygons
    for n, items in enumerate(polys.items()):
        name, Pol = items
        if n == 0:
            union = Pol
         # get maximum extent
        union = union.union(Pol)
        style = {'color':'powderblue', 'alpha':0.4}
        # shade the union/intersection of the polygons
        if overlapping:
            x, y = Pol.exterior.xy
            if name == "selection":
                ax.fill(x, y, **style, zorder=5)
                continue
            ax.plot(x, y, label=name)
        # shade the areas individually
        else:
            if name == "selection":
                continue
            x, y = Pol.exterior.xy
            ax.fill(x, y, **style, zorder=6)
            ax.plot(x, y, label=name, zorder=6)
    # add reference points to the figure
    if ref_points is not None:
        if overlapping and intersection_extent is not None:
            minlon, maxlon, minlat, maxlat = intersection_extent
            mask = (ref_points[:,0]>=minlon) & (ref_points[:,0]<=maxlon) &\
                   (ref_points[:,1]>=minlat) & (ref_points[:,1]<=maxlat)
            selected = ref_points[mask]
            outside = ref_points[~ mask]
        else:
            selected, outside = ref_points, np.array([])
        marker_styles = [
            {"marker": "o", "c":"turquoise", "s":15},
            {"marker": "o", "c":"tomato", "s":15},
        ]
        # mapplot with imshow for gridded (non-ISMN) references
        if reg_grid:
            plot_df = []
            for n, (point_set, style, name) in enumerate(zip(
                    (selected, outside),
                    marker_styles,
                    ("Selected reference validation points", "Validation points outside selection")
            )):
                if point_set.size != 0:
                    point_set = point_set.transpose()
                    index = pd.MultiIndex.from_arrays(point_set, names=('lon', 'lat'))
                    point_set = pd.Series(
                        data=n,
                        index=index,
                    )
                    plot_df.append(point_set)
                    # plot point to 'fake' legend entry
                    ax.scatter(0, 0, label=name, marker="s", s=10, c=style["c"])
                else:
                    continue
            plot_df = pd.concat(plot_df, axis=0)
            zz, zz_extent, origin = geotraj_to_geo2d(
                plot_df,
                grid_stepsize=grid_stepsize
            )
            cmap = mcol.LinearSegmentedColormap.from_list('mycmap', ['turquoise', 'tomato'])
            im = ax.imshow(
                zz, cmap=cmap,
                origin=origin, extent=zz_extent,
                transform=globals.data_crs, zorder=4
            )
        # scatterplot for ISMN reference
        else:
            for point_set, style, name in zip(
                    (selected, outside),
                    marker_styles,
                    ("Selected reference validation points", "Validation points outside selection")
            ):
                if point_set.size != 0:
                    im = ax.scatter(
                        point_set[:,0], point_set[:,1],
                        edgecolors='black', linewidths=0.1,
                        zorder=4, transform=globals.data_crs,
                        **style, label=name
                    )
                else:
                    continue
    # style plot
    make_watermark(fig, globals.watermark_pos, offset=0)
    title_style = {"fontsize": 12}
    ax.set_title("Spatial extent of the comparison", **title_style)
    # provide extent of plot
    d_lon = abs(union.bounds[0] - union.bounds[2])* 1/8
    d_lat = abs(union.bounds[1] - union.bounds[3])* 1/8
    plot_extent = (union.bounds[0] - d_lon, union.bounds[2] + d_lon,
                   union.bounds[1] - d_lat, union.bounds[3] + d_lat)
    grid_intervals = [1, 5, 10, 30]
    style_map(ax, plot_extent, grid_intervals=grid_intervals)
    # create legend
    plt.legend(
        loc="lower left",
        fontsize='small',
        framealpha=0.4,
        edgecolor='black'
    )
