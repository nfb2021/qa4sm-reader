# -*- coding: utf-8 -*-
import os
import pandas as pd
from qa4sm_reader.plotter import QA4SMPlotter
from qa4sm_reader.img import QA4SMImg
from qa4sm_reader import globals
import matplotlib.pyplot as plt

def plot_all(filepath:str,
             metrics:list=None,
             extent:tuple=None,
             out_dir:str=None,
             out_types:str='png',
             save_all:bool=True,
             **plotting_kwargs) -> (list, list):
    """
    Creates boxplots for all metrics and map plots for all variables.
    Saves the output in a folder-structure.

    Parameters
    ----------
    filepath : str
        path to the *.nc file to be processed.
    metrics : set or list, optional (default: None)
        metrics to be plotted. If None, all metrics with data are plotted
    extent : tuple, optional (default: None)
        Area to subset the values for -> (min_lon, max_lon, min_lat, max_lat)
    out_dir : str, optional (default: None)
        Path to output generated plot. If None, defaults to the current working directory.
    out_types: str or list
        extensions which the files should be saved in
    save_all: bool, optional. Default is True.
        all plotted images are saved to the output directory
    plotting_kwargs: arguments for plotting functions.

    Returns
    -------
    fnames_boxplots: list
    fnames_mapplots: list
        lists of filenames for created mapplots and boxplots
    """
    # initialise image and plotter
    img = QA4SMImg(filepath, extent=extent, ignore_empty=True)
    plotter = QA4SMPlotter(image=img, out_dir=out_dir)

    if not metrics:
        metrics = img.metrics
    # iterate metrics and create files in output directory
    fnames_bplot, fnames_mapplot = [], []
    for metric in metrics:
        metric_bplots, metric_mapplots = plotter.plot_metric(metric=metric,
                                                             out_types=out_types,
                                                             save_all=save_all,
                                                             **plotting_kwargs)
        # there can be boxplots with no mapplots
        if metric_bplots:
            fnames_bplot.extend(metric_bplots)
        if metric_mapplots:
            fnames_mapplot.extend(metric_mapplots)
        
    return fnames_bplot, fnames_mapplot

def get_img_stats(filepath:str, extent:tuple=None) -> pd.DataFrame:
    """
    Creates a dataframe containing summary statistics for each metric

    Parameters
    ----------
    filepath : str
        path to the *.nc file to be processed.
    extent : list
        list(x_min, x_max, y_min, y_max) to create a subset of the values

    Returns
    -------
    table : pd.DataFrame
        Quick inspection table of the results.
    """
    img = QA4SMImg(filepath, extent=extent, ignore_empty=True)
    table = img.stats_df()
    
    return table

nc = '/home/pstradio/Projects/scratch/Difference_plot_data/0-ISMN.soil moisture_with_1-ESA_CCI_SM_active.sm_with_2-GLDAS.SoilMoi0_10cm_inst.nc'
out = '/home/pstradio/Projects/scratch/Difference_plot_data/out1'

from qa4sm_reader.img import QA4SMImg

im = QA4SMImg(nc)
plot_all(nc, out_dir=out)
get_img_stats(nc)
