from qa4sm_reader.img import QA4SMImg
from qa4sm_reader.plot_utils import diff_plot, mapplot, boxplot
from qa4sm_reader.handlers import QA4SMDatasets, QA4SMMetricVariable, QA4SMMetric
import qa4sm_reader.globals as glob

from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import warnings as warn


class ComparisonException(Exception):  #todo: migrate plotting functions to utils
    pass


class QA4SMComparison(): # todo: add comparison of spatial extents
    """
    Class that provides comparison plots and table for a list of netCDF files. As initialising a QA4SMImage can
    take some time, the class can be updated keeping memory of what has already been initialized
    """
    def __init__(self, paths:list, extent:tuple=None, where:str='union'):
        """
        Initialise the QA4SMImages and creates a default comparison

        Parameters
        ----------
        paths : list
            list of paths to .nc validation results files to use for the comparison
        extent : tuple, optional (default: None)
            Area to subset the values for.
            (min_lon, max_lon, min_lat, max_lat)
        where : str, optional. Default is 'union'.
            if extent is not specified, we can either take the union or intersection of the original extents of the
            passed .nc files. This affects the diff_table and diff_boxplot methods, whereas the diff_corr, diff_mapplot
            and diff_plot ALWAYS consider the intersection.

        Attributes
        ----------
        comparison : dict
            dictionary with shape {path: (id, QA4SMImage)}
        ref : tuple
            QA4SMImage for the reference validation
        """
        self.paths = paths

        self.comparison = self._init_imgs(paths=paths, extent=extent, where=where)
        self.ref = self._check_ref()

    def _check_ref(self):
        """ Check that all initialized validation results have the same dataset as reference """
        for id, img in self.comparison.values():
            ref = img.datasets._dc_names(img.datasets._ref_dc())
            if id != 0:
                assert ref == previous, "The initialized validation results have different reference datasets. " \
                                        "This is currently not supported"
            previous = ref

        return ref

    @staticmethod
    def _check_extent(extents:list, where:str='union') -> tuple:
        """
        Check if different spatial subsets are overlapping and return either their intersection or union.

        Parameters
        ----------
        extent : list
            list with the spatial extent
        where : str, optional. Default is 'union'.
            if extent is not specified, we can either take the union or intersection of the original extents of the
            passed .nc files. Possible choices are 'union', 'intersection'

        Return
        ------
        extent: tuple
            spatial extent deriving from union or intersection of inputs
        """
        polys = []
        for minlon, maxlon, minlat, maxlat in extents:
            bounds = [(minlon,minlat), (maxlon, minlat),
                      (maxlon, maxlat), (minlon, maxlat)]
            poly = Polygon(bounds)
            polys.append(poly)

        output = polys[0]  # define starting point
        for poly in polys:
            if where == 'union':  # get maximum extent
                output = output.union(poly)
            if where == 'intersection':  # get maximum common
                output = output.intersection(poly)
                assert output, "The spatial extents of the chosen validation results do not overlap. " \
                               "Set 'where' to 'union' to perform the comparison."

        minlon, minlat, maxlon, maxlat = output.bounds

        return minlon, maxlon, minlat, maxlat

    def _init_imgs(self, paths:list=None, extent:tuple=None, where:str='union') -> dict:
        """
        Initialize the QA4SMImages for the selected validation results files.

        Returns
        -------
        comparison: dict
            see self.comparison
        """
        comparison = {}
        subsets = []
        try:
            for n, path in enumerate(paths):
                img = QA4SMImg(path, extent=extent)
                comparison[path] = (n, img)
                subsets.append(img.extent)

        except AssertionError as e:
            e.message = "One of the initialised validation result files has no points in the given spatial subset:" \
                        "{}. \nYou should change subset to a valid one, or not pass any.".format(extent)
            raise e

        if not extent:
            extent = self._check_extent(subsets, where=where)
            for n, path in enumerate(paths):
                img = QA4SMImg(path, extent=extent)
                comparison[path] = (n, img)

        self.extent = extent

        return comparison

    def _subset_comparison(self, subset_paths: list):
        """
        Return a subset of the comparison based on the provided paths

        Parameters
        ----------
        subset_paths: list
            list of selected paths
        """
        if subset_paths == self.paths:
            return self.paths

        self._check_initialized(subset_paths)

        subset = {path: self.comparison[path] for path in subset_paths}

        return subset

    def _check_initialized(self, paths):
        """
        Check that the given path has been initialized in the class

        Parameters
        ----------
        paths: str or list
            path(s) to be checked

        Raise
        -----
        KeyError : if path has not been initialized
        """
        hasnot = False
        if isinstance(paths, list):
            for path in paths:
                if not path in self.comparison.keys():
                    hasnot = True
                    break

        elif paths not in self.comparison.keys():
            path = paths
            hasnot = True

        if hasnot:
            raise KeyError ("The path {} has not been initialized in the QA4SMComparison class".format(path))

    def _check_pairwise(self):
        """
        Checks that the current initialized supports pairwise comparison methods

        Raise
        -----
        ComparisonException : if not
        """
        pairwise = True
        if len(list(self.comparison.keys())) > 2:
            pairwise = False

        else:
            for id, img in self.comparison.values():
                if img.datasets.n_datasets() > 2:
                    pairwise = False

        if not pairwise:
            raise ComparisonException (
                "For pairwise comparison methods, only two validation results with two datasets each can be compared")

#----------------------- handling functions ----------------------------------------------------------------------------

# functions here should make changes necessary to perform comparisons (e.g. same gridpoints)

    def match_references(self):
        """Function to match the points of different validations falling in the same spatial subset"""
        # todo: implement

# ---------------------- plotting functions ----------------------------------------------------------------------------

    def _title_plot(self):
        """ Create title for general plot """
        parts = []
        for path in self.paths:
            self._check_initialized(path)
            id, img = self.comparison[path]
            img_part = "{}: {}".format(id, img.name)
            parts.append(img_part)
        title = "Comparison between " + " and ".join(parts)

        return title

    def _get_pairwise(self, metric) -> (list, list):
        """
        Get the data and names for pairwise comparisons, meaning: two validations with one satellite dataset each

        Parameters
        ----------
        metric: str
            name of metric to get data on
        """
        to_plot, names = [], []
        for id, img in self.comparison.values():
            Var = img.group_vars(**{'metric':metric})[0]
            # below necessary to workaround identical variable names
            names.append("{}: \n".format(id) + Var.pretty_name)
            df = Var.values
            to_plot.append(df)

        return to_plot, names

    def diff_table(self) -> pd.DataFrame:
        """
        Create a table where all the metrics for the different validation runs are compared
        """
        self._check_pairwise() # todo: handle other cases

        medians, names = [], []
        for id, img in self.comparison.values():
            median = img.stats_df()['Median']
            medians.append(median)
            names.append("Medians for {}".format(img.name))

        table = pd.concat(medians, axis=1)
        table.columns = names
        table['Difference of medians (source - reference)'] = table[names[1]] - table[names[0]]
        pd.set_option('display.precision', 1) # todo: format numbers

        return table

    def diff_bplot_extended(self, metric:str):
        """
        Create a boxplot where two or more validations are compared
        """
        self._check_pairwise() # todo: handle other cases
        # get data and names
        to_plot, names = self._get_pairwise(metric=metric)
        # prepare names
        Metric = QA4SMMetric(metric)
        ref_ds = self.ref['short_name']
        um = glob._metric_description[metric].format(glob._metric_units[ref_ds])
        # prepare data
        boxplot_df = pd.concat(to_plot, axis=1)
        boxplot_df.columns = names
        boxplot_df["Difference (source - reference)"] = boxplot_df[names[1]] - boxplot_df[names[0]]
        palette = sns.color_palette(palette=['white', 'white', 'pink'], n_colors=3)
        # plot data
        fig, axes = boxplot(boxplot_df,
                            label= "{} {}".format(Metric.pretty_name, um),
                            figsize=(16,10),
                            **{'palette':palette})
        axes.set_title(self._title_plot())

    def diff_boxplot(self, metric:str):
        """
        Create a boxplot where two or more validations are compared
        """
        # get non-pairwise data to plot
        to_plot, names = [], []
        for n, res in enumerate(self.comparison.values()):
            id, img = res
            for Var in img._iter_vars(**{'metric':metric}):
                names.append("{}: \n".format(id) + Var.pretty_name)
                # below necessary to workaround identical variable names
                df = Var.values
                df.columns = [str(n) + Var.varname]
                to_plot.append(df)

        Metric = QA4SMMetric(metric)
        ref_ds = self.ref['short_name']
        um = glob._metric_description[metric].format(glob._metric_units[ref_ds])
        # prepare data
        boxplot_df = pd.concat(to_plot, axis=1)
        boxplot_df.columns = names
        # plot data
        fig, axes = boxplot(boxplot_df, label= "{} {}".format(Metric.pretty_name, um), figsize=(16,10))
        axes.set_title(self._title_plot())

    def diff_plot(self, metric:str, **kwargs):
        """
        Create a Bland Altman plot where two or more validations can be compared, for a metric. Difference is other - reference

        Parameters
        ----------
        ref : str
            path to the reference validation result .nc file
        others : list
            list of paths to the validation results files to be compared tot he reference
        metric: str
            metric from the .nc result file attributes that the plot is based on
        **kwargs : kwargs
            plotting keyword arguments
        """
        other_dfs, other_names = [], []
        for id, img in self.comparison.values():
            for n, Var in enumerate(img._iter_vars(**{'metric':metric})):
                name = "{}: ".format(id) + Var.pretty_name
                if id == 0 and n == 0:
                    ref_img = img # todo: unit measures for comparisons between different datasets
                    ref_df = Var.values
                    ref_name = name
                else:
                    other_dfs.append(Var.values)
                    other_names.append(name)

        fig, axes = diff_plot(ref_df, other_dfs, ref_name, other_names, )
        # get unit measures
        ref_ds = self.ref['short_name']
        um = glob._metric_description[metric].format(glob._metric_units[ref_ds])
        # set figure title
        Metric = QA4SMMetric(metric)
        axes.set_title(self._title_plot() + " for {}".format(Metric.pretty_name))
        # set axes titles
        axes.set_xlabel('Mean with {}'.format(ref_name) + um)
        axes.set_ylabel('Difference with {}'.format(ref_name) + um)

    def corr_plot(self, metric:str, **sns_kwargs):
        """
        Correlation plot between two validation results, for a metric
        """
        self._check_pairwise() # todo: handle other cases
        # get data and names
        to_plot, names = self._get_pairwise(metric=metric)
        # get names right
        Metric = QA4SMMetric(metric)
        ref_ds = self.ref['short_name']
        um = glob._metric_description[metric].format(glob._metric_units[ref_ds])
        corrplot_df = pd.concat(to_plot, axis=1)
        corrplot_df.columns = names
        # plot regression
        slope, int, r, p, sterr = stats.linregress(corrplot_df[names[0]], corrplot_df[names[1]])
        fig, ax = plt.subplots(figsize=(16,10))
        ax = sns.regplot(x=corrplot_df[names[0]],
                         y=corrplot_df[names[1]],
                         label="Validation point",
                         line_kws={'label':"x ={}*y + {}, r: {}, p: {}".format(
                             *[round(i,2) for i in [slope, int, r, p]])},
                         **sns_kwargs)
        ax.set_title(self._title_plot() + " for {}".format(Metric.pretty_name))
        plt.legend()

    def diff_mapplot(self, metric:str, diff_range:str='adjusted'):
        """
        Create a pairwise mapplot of the difference between the validations, for a metric. Difference is other - reference

        Parameters
        ----------
        diff_range: str, default is 'adjusted'
            if 'adjusted', colorbar goues from minimum to maximum of difference; if 'fixed', the colorbar goes from the
            maximum to the minimum difference range, by metric
        """
        self._check_pairwise() # todo: handle other cases
        Metric = QA4SMMetric(metric)

        sets = []
        for id, img in self.comparison.values():
            for Var in img._iter_vars(**{'metric':metric}):
                sets.append(Var.values)
        # get difference (other - reference) values for mapplot
        assert len(sets) == 2
        diff = pd.concat(sets, axis=1).diff(axis=1)
        diff.columns = ['nan', 'difference']
        cbar_label = "Difference of  {}".format(Metric.pretty_name)
        # make mapplot
        fig, axes = mapplot(diff['difference'],
                            metric,
                            self.ref['short_name'],
                            diff_range=diff_range,
                            label=cbar_label)
        axes.set_title(self._title_plot() + " for {}".format(Metric.pretty_name))

    def diff_method(self, method):
        """
        Return the difference function from a lookup table
        """
        try:
            diff_methods_lut = {'table': self.diff_plot,
                                'boxplot': self.diff_boxplot,
                                'extended_bplot': self.diff_bplot_extended,
                                'correlation_plot': self.corr_plot,
                                'plot': self.diff_plot,
                                'mapplot': self.diff_mapplot}
        except IndexError as e:
            warn('Difference method not valid. Choose one of %s' % ', '.join(diff_methods_lut.keys()))
            raise e

        return diff_methods_lut[method]

    def wrapper(self, paths, method): # todo: make functional
        """
        Call the method using a list of paths and the already initialised images

        Properties
        ----------
        paths: list
            list of paths to .nc validation result files
        method: str
            a method from the lookup table in diff_method
        """
        subset = self._subset_comparison(paths)
        subset_comparison = [value[1] for value in subset.values()]

        diff_funct = self.diff_methods(method)
        output = diff_funct(self.ref, comparison)

        return output
