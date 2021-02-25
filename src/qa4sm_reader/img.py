# -*- coding: utf-8 -*-

import xarray as xr
from qa4sm_reader import globals
from parse import *
import os
import numpy as np
from collections import OrderedDict
from qa4sm_reader.handlers import _build_fname_templ
from qa4sm_reader.handlers import QA4SMMetricVariable
import pandas as pd
import itertools


class QA4SMImg(object):
    """
    A QA4SM validation results netcdf image.
    """
    def __init__(self, filepath, extent=None, ignore_empty=True, metrics=None,
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

        self.common, self.double, self.triple = self._load_metrics_from_file(metrics)

        self.ref_dataset = self.ds.val_dc_dataset0
        # this try here is to obey tests, withouth a necessity of changing and commiting test files again
        try:
            self.ref_dataset_grid_stepsize = self.ds.val_dc_dataset0_grid_stepsize
        except:
            self.ref_dataset_grid_stepsize = 'nan'

    def _load_metrics_from_file(self, metrics:list=None) -> (dict, dict, dict):
        """ Load and group all metrics from file """
        self.df = self._ds2df(None)

        common, double, triple = dict(), dict(), dict()
        if metrics is None:
            metrics = list(itertools.chain(*list(globals.metric_groups.values())))
        for metric in metrics:
            # todo: loading every single variable is slow
            metr_vars = self._load_metric_from_file(metric)
            if len(metr_vars) > 0:
                if metric in globals.metric_groups[2]:
                    double[metric] = metr_vars
                elif metric in globals.metric_groups[3]:
                    triple[metric] = metr_vars
                else:
                    common[metric] = metr_vars

        return common, double, triple

    def _load_metric_from_file(self, metric:str) -> np.array:
        """ Load all variables that describe the metric from file. """

        all_vars = np.sort(np.array(list(self.ds.variables.keys())))
        metr_vars = []
        for var in all_vars:
            Var = self._load_var(var, empty=True)
            if Var is not None and (Var.metric == metric):
                Var.values = self.df[[var]].dropna()
                if self.ignore_empty:
                    if not Var.isempty():
                        metr_vars.append(Var)
                else:
                    metr_vars.append(Var)

        return np.array(metr_vars)

    def _load_var(self, varname:str, empty=False) -> (QA4SMMetricVariable or None):
        """ Create a common variable and fill it with values """
        if empty:
            values = None
        else:
            values = self.df[[varname]]
        try:
            Var = QA4SMMetricVariable(varname, self.ds.attrs, values=values)
            return Var
        except IOError:
            return None


    def _ds2df(self, varnames:list=None) -> pd.DataFrame:
        """ Cut a variable to extent and return it as a values frame """
        try:
            if varnames is None:
                if globals.time_name in list(self.ds.variables.keys()):
                    if len(self.ds[globals.time_name]) == 0:
                        self.ds = self.ds.drop_vars('time')
                df = self.ds.to_dataframe()
            else:
                df = self.ds[self.index_names + varnames].to_dataframe()
                df.dropna(axis='index', subset=varnames, inplace=True)
        except KeyError as e:
            raise Exception(
                'The given variable ' + ', '.join(varnames) +
                ' do not match the names in the input values.' + str(e))

        if isinstance(df.index, pd.MultiIndex):
            lat, lon = globals.index_names
            df[lat] = df.index.get_level_values(lat)
            df[lon] = df.index.get_level_values(lon)

        if self.extent:  # === geographical subset ===
            lat, lon = globals.index_names
            df = df[(df[lon] >= self.extent[0]) & (df[lon] <= self.extent[1]) &
                    (df[lat] >= self.extent[2]) & (df[lat] <= self.extent[3])]

        df.reset_index(drop=True, inplace=True)
        df = df.set_index(self.index_names)

        return df

    def metric_df(self, metric):
        """
        Group all variables for the metric in a common data frame

        Parameters
        ---------
        metric : str
            The name of a metric in the file, all variables for that metric are
            combined into one values frame.

        Returns
        -------
        df : pd.DataFrame
            A dataframe that contains all variables that describe the metric
            in the column
        """
        for g, metric_group in {0: self.common, 2: self.double, 3: self.triple}.items():
            if metric in metric_group.keys():
                if g != 3:
                    conc = [Var.values for Var in metric_group[metric]]
                    return pd.concat(conc, axis=1)
                else:
                    mds_df = {}
                    for Var in metric_group[metric]:
                        _, _, mds_meta = Var.get_varmeta()
                        k = (mds_meta[0], mds_meta[1]['short_name'], mds_meta[1]['short_version'])
                        if k not in mds_df.keys():
                            mds_df[k] = [Var.values]
                        else:
                            mds_df[k].append(Var.values)
                    ret = []
                    for k, dflist in mds_df.items():
                        try:
                            r = pd.concat(dflist, sort=True, axis=1)
                        except ValueError:
                            r = pd.concat(dflist, sort=True, axis=0)
                        ret.append(r)
                    return ret

    def find_group(self, src):
        """
        Search the element and get the variable group that it is in.

        Parameters
        ---------
        src : str
            Either a metric or a variable

        Returns
        -------
        metric_group : dict
            A collection of metrics for 2, 3 or all datasets.
        """
        for metric_group in [self.common, self.double, self.triple]:
            if src in metric_group.keys():
                return metric_group
        for metric_group in [self.common, self.double, self.triple]:
            for metric in metric_group.keys():
                if src in [Var.varname for Var in metric_group[metric]]:
                    return metric_group

    def ref_meta(self) -> tuple:
        """ Go through all variables and check if the reference dataset is the same """
        ref_meta = None
        for metric_group in [self.common, self.double, self.triple]:
            for metric, vars in metric_group.items():
                for Var in vars:
                    if ref_meta is None:
                        ref_meta, _, _ = Var.get_varmeta()
                    else:
                        new_ref_meta, _, _ = Var.get_varmeta()
                        assert new_ref_meta == ref_meta
        return ref_meta

    def var_meta(self, varname):
        """
        Get the metric and metadata for a single variable.

        Parameters
        --------
        varname : str
            The variable that is looked up

        Returns
        -------
        var_meta : dict
            metric as the key and ref_meta, dss_meta and mds_meta as the
            values.
        """

        metr_group = self.find_group(varname)
        for metric, vars in metr_group.items():
            for Var in vars:
                if Var.varname == varname:
                    return {Var.metric: Var.get_varmeta()}


    def metric_meta(self, metric):
        """
        Get the meta values for all variables that describe the passed metric.

        Parameters
        ----------
        metric : str
            A metric that is in the file (e.g. n_obs, R, ...)

        Returns
        -------
        metric_meta : dict
            Dictionary of metadata dictionaries, with variables as the keys.
        """

        group = self.find_group(metric)
        metvar_meta = {}
        for Var in group[metric]:
            metvar_meta[Var.varname] = Var.get_varmeta()
        return metvar_meta
            
    def parse_filename(self):
        """
        Parse filename and derive the validation datasets. Relies on the separator
        between values sets in the filename from the globals.

        Parameters
        ----------
        filename : str
            File name (not path) to parse based on the rule from globals.

        Returns
        -------
        ds_and_vers : dict
            The parsed datasets and version from the file name.
        """
        filename = os.path.basename(self.filepath)
        parts = filename.split(globals.ds_fn_sep)
        fname_templ = _build_fname_templ(len(parts))
        return parse(fname_templ, filename).named

    def ls_metrics(self, as_groups=True):
        """
        Get a list of validation METRICS names that are in the current loaded.

        Parameters
        ----------
        as_groups : bool, optional (default: True)
            Return the metrics grouped by whether all datasets, two, or three
            are considered during calculation. If this is False, then all metrics
            are combined in a common list

        Returns
        -------
        metrics : list or OrderedDict
            The (grouped) metrics in the current file.
        """

        common = [m for m in self.common.keys()]
        double = [m for m in self.double.keys()]
        triple = [m for m in self.triple.keys()]

        if as_groups:
            return OrderedDict([('common', common), ('double', double),
                                ('triple', triple)])
        else:
            return np.sort(np.array(common + double + triple))

    def ls_vars(self, as_groups=True):
        """
        Get a list of VARIABLES (except gpi, lon, lat) of the current file.

        Parameters
        ---------
        only_metrics_vars : bool
            Only include variables that describe a validation metric
        exclude_empty : bool, optional (default: True)
            Ignore variables that are empty / all nans from reading

        Returns
        --------
        vars : np.array
            Alphabetically sorted variables in the file (except gpi, lon, lat),
            optionally without the empty variables.
        """
        common, double, triple = None, None, None
        for g, metric_group in zip((0,2,3), (self.common, self.double, self.triple)):
            varnames = []
            for metric in metric_group.keys():
                for Var in metric_group[metric]:
                    varnames.append(Var.varname)
            if g == 0:
                common = varnames
            elif g == 2:
                double = varnames
            elif g == 3:
                triple = varnames
            else:
                raise NotImplementedError

        if as_groups:
            return OrderedDict([('common', common), ('double', double),
                                ('triple', triple)])
        else:
            return np.sort(np.array(common + double + triple))
    
    def metric_stats(self, metric):
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
        group = self.find_group(metric)
        metric_vars = group[metric]
        metric_stats = []
        # check whether more than 2 datasets are compared to reference
        three_or_more = len(metric_vars) > 2
        # for all the variables in a metric
        for n, metric_var in enumerate(metric_vars):
            ref_meta, dss_meta, mds_meta = metric_var.get_varmeta()
            # get interquartile range 
            values = metric_var.values
            iqr = values.quantile(q=[0.75,0.25]).diff()
            iqr = abs(float(iqr.loc[0.25]))
            # find the statistics for the metric variable
            if metric_var.g == 0:
                var_stats = [round(float(i),1) for i in (values.mean(), values.median(), iqr)]
                var_stats.append('All datasets')
                var_stats.extend([globals._metric_name[metric], metric_var.g])

            else:
                var_stats = [np.format_float_scientific(float(i), 2) for i in (values.mean(), values.median(), iqr)]

                if metric_var.g == 2:
                    i, ds_name = dss_meta[0]
                    var_stats.append('{}-{} ({})'.format(i, ds_name['short_name'], ds_name['pretty_version']))

                elif metric_var.g == 3:
                    i, ds_name = mds_meta

                    if not three_or_more:
                        var_stats.append('{}-{} ({})'.format(i, ds_name['short_name'], ds_name['pretty_version']))

                    else:
                        for ds in dss_meta:
                            if not ds == mds_meta:
                                o, other_ds  = ds

                                break

                        var_stats.append('{}-{} ({}); other ref: {}-{} ({})'.format(i, ds_name['short_name'],
                                                                                    ds_name['pretty_version'],
                                                                                    o, other_ds['short_name'],
                                                                                    other_ds['pretty_version']))

                var_stats.extend([globals._metric_name[metric] + globals._metric_description_HTML[metric].format(
                    globals._metric_units_HTML[ds_name['short_name']]), metric_var.g])
            # put the separate variable statistics in the same list
            metric_stats.append(var_stats)
        
        return metric_stats
    
    def stats_df(self):
        """
        Create a DataFrame with summary statistics for all the metrics

        Returns
        -------
        stats_df : pd.DataFrame
            Quick inspection table of the results.
        """
        stats = []
        # find stats for all the metrics
        for metric in self.ls_metrics(False):
            stats.extend(self.metric_stats(metric))
        # create a dataframe
        stats_df = pd.DataFrame(stats, columns = ['Mean', 'Median', 'IQ range', 'Dataset', 'Metric', 'Group'])
        stats_df.set_index('Metric', inplace=True)
        stats_df.sort_values(by='Group', inplace=True)
        
        return stats_df
