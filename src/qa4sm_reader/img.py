# -*- coding: utf-8 -*-

import xarray as xr
from qa4sm_reader import globals
from parse import *
import os
import numpy as np
from collections import OrderedDict
from qa4sm_reader.handlers import QA4SMDatasets, QA4SMMetricVariable, QA4SMMetric
import pandas as pd
import itertools


class QA4SMImg():
    """
    A QA4SM validation results netcdf image.
    """
    def __init__(self, filepath,
                 extent=None,
                 ignore_empty=True,
                 metrics=None,
                 index_names=globals.index_names):
        """
        Initialise a common QA4SM results image.

        Parameters
        ----------
        filepath : str
            Path to the results netcdf file (as created by QA4SM)
        extent : tuple, optional (default: None)
            Area to subset the values for.
            (min_lon, max_lon, min_lat, max_lat)
        ignore_empty : bool, optional (default: True)
            Ignore empty variables in the file.
        metrics : list or None, optional (default: None)
            Subset of the metrics to load from file, if None are passed, all
            are loaded.
        index_names : list, optional (default: ['lat', 'lon'] - as in globals.py)
            Names of dimension variables in x and y direction (lat, lon).
        """
        self.filepath = filepath
        self.filename = os.path.basename(self.filepath)

        self.extent = extent
        self.index_names = index_names

        self.ignore_empty = ignore_empty
        self.ds = xr.open_dataset(self.filepath)
        self.varnames = list(self.ds.variables.keys())

        self.df = self._ds2df()
        self.datasets = QA4SMDatasets(self.ds.attrs)
        self.vars = self._load_vars()
        self.metrics = self._load_metrics()
        self.common, self.double, self.triple = self.group_metrics(metrics)

        self.ref_dataset = self.ds.val_dc_dataset0
        self.name = self.create_image_name()

        try:
            self.ref_dataset_grid_stepsize = self.ds.val_dc_dataset0_grid_stepsize
        except:
            self.ref_dataset_grid_stepsize = 'nan'
        # todo: update tests for sel.ds.val_dc_dataset0_grid_stepsize = 'nan'

    def _load_vars(self, empty=False) -> (list, list):
        """
        Create a list of common variables and fill each with values

        Parameters
        ----------
        empty : bool, default is False
            if True, each Var.values is empty

        Returns
        -------
        meta_vars: list
            list of QA4SMMetricVariable objects for the metadata variables (lon, lat, gpi, ...)
        vars : list
            list of QA4SMMetricVariable objects for the validation variables
        """
        vars = []
        for varname in self.varnames:
            if empty or varname in ['lat', 'lon', 'time']: # todo: is there way to not specify names?
                values = None
            else:
                values = self.df[[varname]]

            try:
                Var = QA4SMMetricVariable(varname, self.ds.attrs, values=values)
            except IOError:
                Var = None
                continue

            if not Var is None:
                vars.append(Var)

        return vars

    def _load_metrics(self) -> dict:
        """
        Create a list of metrics for the file

        Returns
        -------
        Metrics : dict
            dictionary with shape {metric name: QA4SMMetric}
        """
        Metrics = {}
        all_groups = globals.metric_groups.values()
        for group in all_groups:
            for metric in group:
                metric_vars = []
                for Var in self._iter_vars(**{'metric': metric}):
                    metric_vars.append(Var)

                if metric_vars != []:
                    Metric = QA4SMMetric(metric, metric_vars)
                    Metrics[metric] = Metric

        return Metrics

    def _iter_vars(self, **filter_parms) -> iter:
        """
        Iter through QA4SMMetricVariable objects that are in the file

        Parameters
        ----------
        **filter_parms : kwargs, dict
            dictionary with QA4SMMetricVariable attributes as keys and filter value as values (e.g. {g: 0})
        """
        for Var in self.vars:
            if filter_parms:
                for key, val in filter_parms.items():
                    if getattr(Var, key) == val:
                        yield Var
            else:
                yield Var

    def _iter_metrics(self, **filter_parms) -> iter: # todo: use Wolfi method instead of kwargs
        """
        Iter through QA4SMMetric objects that are in the file

        Parameters
        ----------
        **filter_parms : kwargs, dict
            dictionary with QA4SMMetric attributes as keys and filter value as values (e.g. {g: 0})
        """
        for Metric in self.metrics.values():
            for key, val in filter_parms.items():
                if val is None or getattr(Metric, key) == val:
                    yield Metric

    def group_vars(self, **filter_parms):
        """
        Return a list of QA4SMMetricVariable that match filters

        Parameters
        ----------
        **filter_parms : kwargs, dict
            dictionary with QA4SMMetricVariable attributes as keys and filter value as values (e.g. {g: 0})
        """
        vars = []
        for Var in self._iter_vars(**filter_parms):
            vars.append(Var)

        return vars

    def group_metrics(self, metrics:list=None) -> (dict, dict, dict):
        """
        Load and group all metrics from file

        Parameters
        ----------
        metrics: list or None
            if list, only metrics in the list are grouped
        """
        common, double, triple = dict(), dict(), dict()

        # fetch Metrics
        if metrics is None:
            metrics = self.metrics.keys()

        # fill dictionaries
        for metric in metrics:
            Metric = self.metrics[metric]
            if Metric.g == 0:
                common[metric] = Metric
            elif Metric.g == 2:
                double[metric] = Metric
            elif Metric.g == 3:
                triple[metric] = Metric

        return common, double, triple

    def _ds2df(self, varnames:list=None) -> pd.DataFrame:
        """
        Cut a variable to extent and return all variables in a single DataFrame.

        Parameters
        ----------
        varnames : list or None
            list of QA4SMMetricVariables to be placed in the DataFrame

        Return
        ------
        df : pd.DataFrame
            DataFrame with Var name as column names and values cut to self.extent
        """
        try:
            if varnames is None:
                if globals.time_name in self.varnames:
                    if len(self.ds[globals.time_name]) == 0:
                        self.ds = self.ds.drop_vars('time')
                df = self.ds.to_dataframe()
            else:
                df = self.ds[self.index_names + varnames].to_dataframe()
                df.dropna(axis='index', subset=varnames, inplace=True)
        except KeyError as e:
            raise Exception("The variable name '{}' does not match any name in the input values.".format(e.args[0]))

        if isinstance(df.index, pd.MultiIndex):
            lat, lon = globals.index_names
            df[lat] = df.index.get_level_values(lat)
            df[lon] = df.index.get_level_values(lon)

        # geographical subset of the results
        if self.extent:
            lat, lon = globals.index_names
            df = df[(df[lon] >= self.extent[0]) & (df[lon] <= self.extent[1]) &
                    (df[lat] >= self.extent[2]) & (df[lat] <= self.extent[3])]

        df.reset_index(drop=True, inplace=True)
        df = df.set_index(self.index_names)

        return df

    def metric_df(self, metrics:str or list):
        """
        Group all variables for the metric in a common data frame

        Parameters
        ---------
        metrics : str or list
            name(s) of the metrics to have in the DataFrame

        Returns
        -------
        df : pd.DataFrame
            A dataframe that contains all variables that describe the metric
            in the column
        """
        if isinstance(metrics, list):
            Vars = []
            for metric in metrics:
                Vars.extend(self.group_vars(**{'metric':metric}))
        else:
            Vars = self.group_vars(**{'metric':metrics})

        varnames = [Var.varname for Var in Vars]
        metrics_df = self._ds2df(varnames=varnames)

        return metrics_df

    # todo: replace find_group() everywhere

    # todo: create handlers.Dataset object with precise/flexible structure

    # todo: ref_meta, metric_meta() and var_meta() functions need to be replaced

    # todo: parse_filename() and _build_fname_templ() have been substituted by QA4SMDatasets class

    # todo: create a unique name for the image (to use in plots)
    def create_image_name(self) -> str:
        """ Create a unique name for the QA4SMImage """
        ref = self.datasets.dataset_metadata(self.datasets._ref_dc())[1]
        ref_part = "{} v ".format(ref['pretty_name'])
        others_part = ", ".join(other['pretty_name'] for other in self.datasets.others)
        date_range = "({} to {})".format(self.ds.val_interval_from[:10], self.ds.val_interval_to[:10])

        return ref_part + others_part + date_range

    def _metric_stats(self, metric)  -> list:
        """
        Provide a list with the metric summary statistics (for each variable)

        Parameters
        ----------
        metric : str
            A metric that is in the file (e.g. n_obs, R, ...)

        Returns
        -------
        metric_stats : list
            List of (variable) lists with summary statistics
        """
        metric_stats = []
        for Var in self._iter_vars(**{'metric':metric}):
            # get interquartile range 
            values = Var.values[Var.varname]
            iqr = values.quantile(q=[0.75,0.25]).diff()
            iqr = abs(float(iqr.loc[0.25]))
            # find the statistics for the metric variable
            var_stats = [i for i in (values.mean(), values.median(), iqr)] #todo: solve decimal problem
            if Var.g == 0:
                var_stats.append('All datasets')
                var_stats.extend([globals._metric_name[metric], Var.g])

            else:
                i, ds_name = Var.metric_ds
                if Var.g == 2:
                    var_stats.append('{}-{} ({})'.format(i, ds_name['short_name'], ds_name['pretty_version']))

                elif Var.g == 3:
                    o, other_ds = Var.other_ds
                    var_stats.append('{}-{} ({}); other ref: {}-{} ({})'.format(i, ds_name['short_name'],
                                                                                ds_name['pretty_version'],
                                                                                o, other_ds['short_name'],
                                                                                other_ds['pretty_version']))

                var_stats.extend([globals._metric_name[metric] + globals._metric_description_HTML[metric].format(
                    globals._metric_units_HTML[ds_name['short_name']]), Var.g])
            # put the separate variable statistics in the same list
            metric_stats.append(var_stats)
        
        return metric_stats
    
    def stats_df(self) -> pd.DataFrame:
        """
        Create a DataFrame with summary statistics for all the metrics

        Returns
        -------
        stats_df : pd.DataFrame
            Quick inspection table of the results.
        """
        stats = []
        # find stats for all the metrics
        for metric in self.metrics.keys(): # todo: check discrepancy with metric names
            stats.extend(self._metric_stats(metric))
        # create a dataframe
        stats_df = pd.DataFrame(stats, columns = ['Mean', 'Median', 'IQ range', 'Dataset', 'Metric', 'Group'])
        stats_df.set_index('Metric', inplace=True)
        stats_df.sort_values(by='Group', inplace=True)
        pd.set_option('display.precision', 1)
        
        return stats_df
