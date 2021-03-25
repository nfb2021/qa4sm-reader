# -*- coding: utf-8 -*-

from qa4sm_reader.img import QA4SMImg
import qa4sm_reader.globals as globals
from pathlib import Path
import seaborn as sns
from qa4sm_reader.plot_utils import *
import pandas as pd


class QA4SMPlotter():
    """
    Class to create image files of plots from the validation results in a QA4SMImage
    """
    def __init__(self, image, out_dir=None):
        """
        Create box plots from results in a qa4sm output file.

        Parameters
        ----------
        image : QA4SMImg
            The results object.
        out_dir : str, optional (default: None)
            Path to output generated plot. If None, defaults to the current working directory.
        """
        self.img = image
        if out_dir:
            self.out_dir = Path(out_dir)  # use directory if specified
            if not self.out_dir.exists:
                self.out_dir.mkdir()  # make if not existing
        else:
            self.out_dir = self.img.filepath.parent  # use default otherwise

        self.ref = image.ref_dataset #todo: check working

    def _standard_filename(self, out_name:str, out_dir:str=None, out_type:str='png') -> Path: # todo: can form be improved?
        """
        Standardized behaviour for filenames: if provided name has extension, it is kept; otherwise, it is saved as
        .png to the specified folder or (if none) to the .nc file folder

        Parameters
        ----------
        out_name : str
            output filename (with or without extension)
        out_dir: str, options. Default is None
            path to desired output directory. If None, path points at the same directory of the .nc file
        out_type : str, optional
            contains file extensions to be plotted. If None, uses 'png'

        Returns
        -------
        outname: pathlib.Path
            correct path of the file
        """
        out_name = Path(out_name)
        # provide output directory
        if not out_dir:
            out_dir = self.out_dir.join(out_name)
        else:
            out_name.with_name(out_dir + out_name.name)
        # provide output file type
        if not out_name.suffix:
            if out_type[0] != '.':
                out_type = '.' + out_type
            out_name.with_suffix(out_type)

        return out_name

    @staticmethod
    def _box_stats(ds:pd.Series, med:bool=True, iqr:bool=True, count:bool=True) -> str:
        """
        Create the metric part with stats of the box (axis) caption

        Parameters
        ----------
        ds: pd.Series
            data on which stats are found
        med: bool
        iqr: bool
        count: bool
            statistics

        Returns
        -------
        stats: str
            caption with summary stats
        """
        # interquartile range
        iqr = ds.quantile(q=[0.75,0.25]).diff()
        iqr = abs(float(iqr.loc[0.25]))

        met_str = []
        if med:
            met_str.append('median: {:.3g}'.format(ds.median()))
        if iqr:
            met_str.append('IQR: {:.3g}'.format(iqr))
        if count:
            met_str.append('N: {:d}'.format(ds.count()))
        stats = '\n'.join(met_str)

        return stats

    @staticmethod
    def _box_caption(dss_meta, tc:bool=False) -> str:
        """
        Create the dataset part of the box (axis) caption

        Parameters
        ----------
        dss_meta: id, dict
            id and dictionary from MetricVariable.get_varmeta(), for the metric dataset (non-tc) or for the other
            satellite dataset (tc)
        tc: bool, default is False
            True if TC. Then, caption starts with "Other Data:"

        Returns
        -------
        capt: str
            box caption
        """
        ds_parts = []
        id, meta = dss_meta
        ds_parts.append('{}-{}\n({})'.format(
            id, meta['pretty_name'], meta['pretty_version']))
        capt = '\n and \n'.join(ds_parts)

        if tc:
            capt = 'Other Data:' + '\n' + ds_part

        return capt

    @staticmethod
    def _get_parts_name(var, type='boxplot_basic'):
        """
        Create parts for title according to the type of plot

        Parameters
        ----------
        var: MetricVar
            variable for a metric
        type: str
            type of plot

        Returns
        -------
        parts: list
            list of parts for title
        """
        parts = []
        ref, mds, other = [meta for meta in var.get_varmeta()]
        if type == 'boxplot_basic':
            parts.append(ref[0])
            parts.extend([ref[1]['pretty_name'], ref[1]['pretty_version']])

        elif type in ['boxplot_tc', 'mapplot_basic', 'mapplot_tc']:
            parts.append(mds[0])
            parts.extend([mds[1]['pretty_name'], mds[1]['pretty_version']])
            parts.append(ref[0])
            parts.extend([ref[1]['pretty_name'], ref[1]['pretty_version']])

            if type == 'mapplot_tc':
                parts.append(other[0])
                parts.extend([other[1]['pretty_name'], other[1]['pretty_version']])

        return parts

    @staticmethod
    def _titles_lut(type):
        """
        Lookup table for plot titles

        Parameters
        ----------
        type: str
            type of plot
        """
        titles = {'boxplot_basic': 'Intercomparison of \n{} \nwith {}-{} ({}) \nas the reference',
                  'boxplot_tc': 'Intercomparison of {} \nfor {}-{} ({}) \nwith {}-{} ({}) \nas the reference',
                  'mapplot_basic': '{} for {}-{} ({}) with {}-{} ({}) as the reference',
                  'mapplot_tc': '{} for {}-{} ({}) with {}-{} ({}) and {}-{} ({}) as the references'}

        try:
            return titles[type]

        except IndexError as e: # todo: test
            e.message = "type '{}' is not in the lookup table".format(type)

            raise e

    def create_title(self, var, metric:str, type:str) -> str:
        """
        Create title of the plot

        Parameters
        ----------
        var: MetricVar
            variable for a metric
        type: str
            type of plot
        """
        parts = [globals._metric_name[metric]]
        parts.extend(self._get_parts_name(var=var, type=type))
        title = self._titles_lut(type=type).format(*parts)

        return title

    @staticmethod
    def _filenames_lut(type):
        """
        Lookup table for file names

        Parameters
        ----------
        type: str
            type of plot
        """
        names = {'boxplot_basic': 'boxplot_{}',
                 'boxplot_tc': 'boxplot_{}_for_{}-{}',
                 'mapplot_common': 'overview_{}',
                 'mapplot_double': 'overview_{}_{}-{}_and_{}-{}',
                 'mapplot_tc': 'overview_{}_for_{}-{}_with_{}-{}_and_{}-{}'}

        try:
            return names[type]

        except IndexError as e: # todo: test
            e.message = "type '{}' is not in the lookup table".format(type)

            raise e

    def create_filename(self, var, metric:str, type:str) -> str:
        """
        Create name of the file

        Parameters
        ----------
        var: MetricVar
            variable for a metric
        type: str
            type of plot
        """
        name = self._filenames_lut(type=type)
        # fetch parts of the name for the variable
        parts = [metric]
        if not type in ['boxplot_basic', 'mapplot_common']:
            ref_meta, mds_meta, other_meta = var.get_varmeta()
            parts.extend([mds_meta[0], mds_meta[1]['short_name'],
                          ref_meta[0], ref_meta[1]['short_name']])
        if type == 'mapplot_tc':
            parts.extend([other_meta[0], other_meta[1]['short_name']])

        return name.format(*parts)

    def _yield_values(self, metric: str, vars=None, add_stats:bool=globals.boxplot_printnumbers) -> pd.DataFrame:
        """
        Get iterable with pandas dataframes for all variables of a metric to plot

        Parameters
        ----------
        metric: str
            metric name
        vars: list or None. Default is None
            list of variables to iterate
        add_stats : bool, optional (default: from globals)
            Add stats of median, iqr and N to the box bottom.

        Yield
        -----
        df: pd.DataFrame
            dataframe with variable values and caption name
        Var: QA4SMMetricVariable
            variable corresponding to the dataframe
        """
        if not vars:
            vars = self.img._iter_vars(**{'metric':metric})

        for n, Var in enumerate(vars):
            ref_meta, mds_meta, other_meta = Var.get_varmeta()
            values = Var.values[Var.varname]
            if Var.g == 0:
                box_cap_ds = 'All datasets'
            else:
                box_cap_ds = self._box_caption(mds_meta)
            if add_stats:
                box_stats = self._box_stats(values)
                box_cap = '{}\n{}'.format(box_cap_ds, box_stats)
            else:
                box_cap = box_cap_ds
            df = pd.DataFrame(values, columns=[box_cap])

            yield df, Var

    def _boxplot_definition(self, metric:str,
                            df:pd.DataFrame,
                            type:str,
                            var=None,
                            watermark_pos=globals.watermark_pos,
                            offset=0.1,
                            **kwargs):
        """
        Define parameters of plot

        Parameters
        ----------
        df: pd.DataFrame
            dataframe to plot
        type: str
            one of _titles_lut
        watermark_pos: str
            position of watermark
        offset: float
            offset of boxplots
        """
        # plot label
        parts = [globals._metric_name[metric]]
        parts.append(globals._metric_description[metric].format(
            globals._metric_units[self.ref]))
        label = "{}{}".format(*parts)
        # generate plot
        figwidth = globals.boxplot_width # todo: check width
        figsize = [figwidth, globals.boxplot_height]
        fig, ax = boxplot(df=df, label=label, figsize=figsize, dpi=globals.dpi)

        # when we only need reference dataset from variables (i.e. is the same):
        if not var:
            for Var in self.img._iter_vars(**{'metric':metric}):
                var = Var
                break
        title = self.create_title(var, metric, type=type)
        ax.set_title(title, pad=globals.title_pad)
        # add watermark
        if globals.watermark_pos not in [None, False]:
            make_watermark(fig, watermark_pos, offset)

        return fig, ax

    def _save_plot(self, out_name, out_dir=None, out_types='png'):
        """
        Save plot with name to self.out_dir

        Parameters
        ----------
        out_name: str
            name of output file
        out_types: str or list
            saves to specified format(s)

        Returns
        -------
        fnames: list
            list of file names with all the extensions
        """
        fnames = []
        if isinstance(out_types, str): # todo: check saving and naming process
            out_types = [out_types]
        for ext in out_types:
            fname = self._standard_filename(out_name, out_dir=out_dir, out_type=ext)
            if fname.exists():
                warnings.warn('Overwriting file {}'.format(fname.name))
            plt.savefig(fname, dpi='figure', bbox_inches='tight')
            fnames.append(fname.name)

        return fnames

    def boxplot_basic(self, metric,
                      out_name=None,
                      out_types='png',
                      out_dir=None,
                      save_file:bool=False, # todo: update with this
                      **kwargs): # todo: check outnmae/out_type
        """
        Creates a boxplot for common and double metrics. Saves a figure and returns Matplotlib fig and ax objects for
        further processing.

        Parameters
        ----------
        metric : str
            metric that is collected from the file for all datasets and combined
            into one plot.
        out_name : [ None | str ], optional
            Name of output file.
            If None, defaults to a name that is generated based on the variables.
        out_type : [ str | list | None ], optional
            The file type, e.g. 'png', 'pdf', 'svg', 'tiff'...
            If list, a plot is saved for each type. If None, no file is saved.
        kwargs: arguments for _boxplot_definition function

        Returns
        -------
        fnames: list
            list of file names with all the extensions
        """
        fnames = []  # list to store all filenames
        values = []
        for df, Var in self._yield_values(metric=metric):
            values.append(df)
        values = pd.concat(values)
        # create plot
        fig, ax = self._boxplot_definition(metric=metric,
                                           df=values,
                                           type='boxplot_basic',
                                           **kwargs)
        if not out_name:
            out_name = self.create_filename(Var, metric, type='boxplot_basic')
        # save or return plotting objects
        if save_file:
            fnames = self._save_plot(out_name, out_dir=out_dir, out_types=out_types)
            plt.close()

            return fnames

        else:
            return fig, ax

    def boxplot_tc(self, metric,
                   out_types='png',
                   out_dir=None,
                   **kwargs):
        """
        Creates a boxplot for TC metrics. Saves a figure and returns Matplotlib fig and ax objects for further processing.

        Parameters
        ----------
        metric : str
            metric that is collected from the file for all datasets and combined
            into one plot.
        out_name : [ None | str ], optional
            Name of output file.
            If None, defaults to a name that is generated based on the variables.
        out_type : [ str | list | None ], optional
            The file type, e.g. 'png', 'pdf', 'svg', 'tiff'...
            If list, a plot is saved for each type. The default is png
        kwargs: arguments for _boxplot_definition function

        Returns
        -------
        fnames: list
            list of file names with all the extensions
        """
        fnames = list()  # list of filenames
        for df, Var in self._yield_values(metric=metric):
            # create plot
            fig, ax = self._boxplot_definition(metric=metric,
                                               df=df,
                                               type='boxplot_tc',
                                               var=Var,
                                               **kwargs)
            # save
            out_name = self.create_filename(Var, metric, type='boxplot_tc')
            fnames = self._save_plot(out_name, out_dir=out_dir, out_types=out_types)
            plt.close()

        return fnames

    def mapplot_var(self, var,
                    out_name=None,
                    out_types='png',
                    out_dir=None,
                    save_file:bool=False, # todo: update with this
                    **plot_kwargs):
        """
        Plots values to a map, using the values as color. Plots a scatterplot for
        ISMN and a image plot for other input values.

        Parameters
        ----------
        var : QA4SMMetricVariab;e
            Var in the image to make the map for.
        out_name : [ None | str ], optional
            Name of output file.
            If None, defaults to a name that is generated based on the variables.
        out_type : [ str | list | None ], optional
            The file type, e.g. 'png', 'pdf', 'svg', 'tiff'...
            If list, a plot is saved for each type. If None, no file is saved.
        **plot_kwargs : dict, optional
            Additional keyword arguments that are passed to dfplot.

        Returns
        -------
        fnames: list
            list of file names with all the extensions
        """
        ref_meta, mds_meta, other_meta = var.get_varmeta()
        metric = var.metric
        ref_grid_stepsize = self.img.ref_dataset_grid_stepsize

        # create mapplot
        fig, ax = mapplot(df=var.values[var.varname],
                          metric=metric,
                          ref_short=ref_meta[1]['short_name'],
                          ref_grid_stepsize=ref_grid_stepsize,
                          plot_extent=self.img.extent,
                          **plot_kwargs)
        # title and plot settings depend on the metric group
        if var.g == 0:
            title = "{} between all datasets".format(globals._metric_name[metric])
            out_name = self.create_filename(var, metric, type='mapplot_common')
        elif var.g == 2:
            title = self.create_title(var=var, metric=metric, type='mapplot_basic')
            out_name = self.create_filename(var, metric, type='mapplot_double')
        else:
            title = self.create_title(var=var, metric=metric, type='mapplot_tc') # todo: check titles are ok with QA4SM
            out_name = self.create_filename(var, metric, type='mapplot_tc')
        # use title for plot, make watermark
        ax.set_title(title, pad=globals.title_pad)
        if globals.watermark_pos not in [None, False]:
            make_watermark(fig, globals.watermark_pos, for_map=True)
        # save file or just return the image
        if save_file:
            fnames = self._save_plot(out_name, out_dir=out_dir, out_types=out_types)
            plt.close('all')

            return fnames

        else:
            return fig, ax

    def mapplot(self, metric, out_types='png', **plot_kwargs):
        """
        Mapplot for all variables for a given metric in the loaded file.

        Parameters
        ----------
        metric : str
            Name of a metric. File is searched for variables for that metric.
        out_type : [ str | list | None ], optional
            The file type, e.g. 'png', 'pdf', 'svg', 'tiff'...
            If list, a plot is saved for each type. If None, no file is saved.
        **kwargs : dict, optional
            Additional keyword arguments that are passed to mapplot_var

        Returns
        -------
        fnames : list
            List of files that were created
        """
        fnames = []
        for Var in self.img._iter_vars(**{'metric':metric}):
            fns = self.mapplot_var(Var,
                                   out_name=None,
                                   out_types=out_types,
                                   **plot_kwargs)
            plt.close('all')
            for fn in fns: fnames.append(fn)

        return fnames

    def plot_metric(self, metric, out_types='png', boxplot_kwargs=dict(), mapplot_kwargs=dict()):
        """
        Plot boxplot or mapplot for a certain metric, according to the metric type

        Parameters
        ----------
        metric: str
            name of the metric
        out_type: str
            extention type for the file to be saved
        boxplot_kwargs : dict, optional
            Additional keyword arguments that are passed to the boxplot function.
        mapplot_kwargs : dict, optional
            Additional keyword arguments that are passed to the mapplot function.
        """
        Metric = self.img.metrics[metric]
        if Metric.g == 0 or Metric.g == 2:
            fnames_bplot = self.boxplot_basic(metric=metric, out_names=out_types, out_types=out_types, **boxplot_kwargs)
        elif Metric.g == 3:
            fnames_bplot = self.boxplot_tc(metric=metric, out_types=out_types, **boxplot_kwargs)

        fnames_mapplot = self.mapplot(metric=metric, out_types=out_types, **mapplot_kwargs)

        return fnames_bplot, fnames_mapplot
