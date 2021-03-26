# -*- coding: utf-8 -*-
import os
import pandas as pd
from qa4sm_reader.plotter import QA4SMPlotter
from qa4sm_reader.img import QA4SMImg
from qa4sm_reader import globals
import matplotlib.pyplot as plt

def plot_all(filepath,
             metrics=None,
             extent=None,
             out_dir=None,
             out_types='png',
             save_all=True,
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
        fnames_bplot.extend(metric_bplots)
        fnames_mapplot.extend(metric_mapplots)
        
    return fnames_bplot, fnames_mapplot

def get_img_stats(filepath, extent=None):
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

path = '/home/pstradio/Projects/scratch/Test_reader'
nc = '/0-C3S.sm_with_1-GLDAS.SoilMoi40_100cm_inst.nc'
out = '/out_2'

plot_all(path+nc, out_dir=path+out)
