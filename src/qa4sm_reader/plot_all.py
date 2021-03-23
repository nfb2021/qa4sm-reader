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
             boxplot_kwargs=dict(),
             mapplot_kwargs=dict()) -> (list, list):
    """
    Creates boxplots for all metrics and map plots for all variables.
    Saves the output in a folder-structure.

    Parameters
    ----------
    filepath : str
        path to the *.nc file to be processed.
    metrics : set or list, optional (default: None)
        metrics to be plotted. If None, all metrics with data are plotted
    extent : list
        list(x_min, x_max, y_min, y_max) to create a subset of the values
    out_dir : str, optional
        Parent directory where to generate the folder structure for all plots.
        If None, defaults to the current working directory.
    out_type: str, optional. Default is 'png'
        File types, e.g. 'png', 'pdf', 'svg', 'tiff'...
    boxplot_kwargs : dict, optional
        Additional keyword arguments that are passed to the boxplot function.
    mapplot_kwargs : dict, optional
        Additional keyword arguments that are passed to the mapplot function.

    Returns
    -------
    fnames_boxplots: list
    fnames_mapplots: list
        lists of filenames for created mapplots and boxplots
    """

    if not out_dir:
        out_dir = os.path.join(os.getcwd(), os.path.basename(filepath))
    # initialise image and plotter
    img = QA4SMImg(filepath, extent=extent, ignore_empty=True)
    plotter = QA4SMPlotter(image=img, out_dir=out_dir)

    if not metrics:
        metrics = img.metrics
    # iterate metrics and create files in output directory
    fnames_boxplots, fnames_mapplots = [], []
    for metric in metrics:
        boxplots, mapplots = plotter.plot_metric(metric=metric,
                                                 out_types=out_types,
                                                 boxplot_kwargs=boxplot_kwargs,
                                                 mapplot_kwargs=mapplot_kwargs)
        plt.close('all')
        fnames_boxplots.extend(boxplots)
        fnames_mapplots.extend(mapplots)
        
    return fnames_boxplots, fnames_mapplots

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

plot_all('/home/pstradio/Projects/scratch/Issue_plotting/0-ISMN.soil moisture_with_1-C3S.sm_with_2-C3S.sm_with_3-GLDAS.SoilMoi0_10cm_inst.nc',out_dir='/home/pstradio/Projects/scratch/Issue_plotting')
