from qa4sm_reader.img import QA4SMImg
from qa4sm_reader.plot_utils import mapplot, boxplot, plot_spatial_extent, _format_floats
from qa4sm_reader.handlers import QA4SMDatasets, QA4SMMetricVariable, QA4SMMetric
import qa4sm_reader.globals as glob
from qa4sm_reader.plotter import QA4SMPlotter

from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from typing import Union
import warnings as warn

class ComparisonError(Exception):
    pass

class SpatialExtentError(Exception):
    pass


class QA4SMComparison():
    """
    Class that provides comparison plots and table for a list of netCDF files. As initialising a QA4SMImage can
    take some time, the class can be updated keeping memory of what has already been initialized
    """
    def __init__(self, paths:list or str, extent:tuple=None, get_intersection:bool=True):
        """
        Initialise the QA4SMImages from the paths to netCDF files specified

        Parameters
        ----------
        paths : list or str
            list of paths or single path to .nc validation results files to use for the comparison
        extent : tuple, optional (default: None)
            Area to subset the values for. At the moment has not been implemented as a choice in the service
            (min_lon, max_lon, min_lat, max_lat)
        get_intersection : bool, default is True
            Whether to get the intersection or union of the two spatial exents

        Attributes
        ----------
        comparison : dict or <QA4SMImg object>
            dictionary with shape {path: (id, QA4SMImage)} or single image
        ref : tuple
            QA4SMImage for the reference validation
        """
        self.paths = paths
        self.extent = extent
        # todo: better distinction between single and double image
        self.comparison = self._init_imgs(extent=extent, get_intersection=get_intersection)
        self.ref = self._check_ref()
        self.union = not get_intersection

    def _init_imgs(self, extent:tuple=None, get_intersection:bool=True) -> dict:
        """
        Initialize the QA4SMImages for the selected validation results files. If 'extent' is specified, this is used. If
        not, by default the intersection of results is taken and the images are initialized with it, unless 'get_union'
        is specified. In this case, only diff_table and diff_boxplots can be created (as only non-pairwise methods).

        Parameters
        ----------
        extent: tuple, optional. Default is None
            exent of stapial subset (minlon, maxlon, minlat, maxlat)
        get_intersection : bool, optional. Default is True.
            if extent is not specified, we can either take the union or intersection of the original extents of the
            passed .nc files. This affects the diff_table and diff_boxplot methods, whereas the diff_corr, diff_mapplot
            and diff_plot ALWAYS consider the intersection.

        Returns
        -------
        comparison : dict or Image
            if more nc files are initialized, a dictionary with shape {path: (id, img)}. Otherwise, the single QASMImg
            initialized for the specified path
        """
        if self.single_image:
            if isinstance(self.paths, list):
                self.paths = self.paths[0]

            img = QA4SMImg(self.paths, extent=extent, empty=True)
            if not len(img.datasets.others) > 1:
                raise ComparisonError("A single validation was initialized, with a single "
                                      "non-reference dataset. You should add another comparison term.")

            return img

        comparison = {}
        imgs = []
        for n, path in enumerate(self.paths):
            if extent:
                try:
                    img = QA4SMImg(path, extent=extent, empty=True)
                    imgs.append(img)
                    if path in comparison.keys():
                        warn(
                            "You are initializing the same validation twice"
                        )

                    comparison[path] = (n, img)
                # todo: not sure if this will show up
                except SpatialExtentError as e:
                    e.message = "One of the initialised validation result files has no points in the given spatial subset:" \
                                "{}. \nYou should change subset to a valid one, or not pass any.".format(extent)
                    raise e
            else:
                self.union = True  # save the state 'union' or 'intersection' to a class attribute
                img = QA4SMImg(path, empty=True)
                imgs.append(img)
                if path in comparison.keys():
                    warn(
                        "You are initializing the same validation twice"
                    )

                comparison[path] = (n, img)

        if get_intersection:
            extent = self._combine_geometry(
                get_intersection=get_intersection,
                imgs=imgs,
            )
            self.extent = extent
            self.union = False

        return comparison

    def init_union(self):
        """Re-initialize the images using the union of spatial extents"""
        self.comparison = self._init_imgs(extent=None, get_intersection=False)
        # make sure the new state is stored in the class attribute
        assert self.union

    def _check_ref(self) -> str:
        """Check that all initialized validation results have the same dataset as reference """
        if self.single_image:
            return self.comparison.datasets.ref

        for id, img in self.comparison.values():
            ref = img.datasets.ref
            if id != 0:
                if not ref == previous:
                    raise ComparisonError("The initialized validation results have different reference "
                                          "datasets. This is currently not supported")
            previous = ref

        return ref

    @property
    def common_metrics(self) -> dict:
        """Get dict {short_name: pretty_name} of metrics that can be used in the comparison"""
        for n, img in enumerate(self._iter_imgs()):
            img_metrics = {}
            for metric in img.metrics:
                # hardcoded because n_obs cannot be compared
                if metric == "n_obs":
                    continue
                img_metrics[metric] = glob._metric_name[metric]
            if n==0:
                common_metrics = img_metrics
                continue
            common_keys = common_metrics.keys() & img_metrics.keys()
            common_metrics = {k: common_metrics[k] for k in common_keys}

        return common_metrics

    @property
    def overlapping(self) -> bool:
        """Return True if the initialised validation results have overlapping spatial extents, else False"""
        if self.single_image:  # one validation is always on the same bounds
            return True

        polys = []
        for img in self._iter_imgs():  # get names and extents for all images
            minlon, maxlon, minlat, maxlat = img.extent
            bounds = [(minlon,minlat), (maxlon, minlat),
                      (maxlon, maxlat), (minlon, maxlat)]
            Pol = Polygon(bounds)
            polys.append(Pol)

        for n, Pol in enumerate(polys):
            if n == 0:
                output = Pol  # define starting point
            output = output.intersection(Pol)

        return output.bounds != ()

    @property
    def single_image(self) -> bool:
        """Whether the initialized image(s) are 1 (double) or 2 (single)"""
        if isinstance(self.paths, str):
            return True
        else:
            return len(self.paths) == 1

    def _iter_imgs(self) -> iter:
        """Iterate over the images in the comparison (regardless of single- or double-images)"""
        if self.single_image:
            yield self.comparison
        else:
            for n, img in self.comparison.values():
                yield img

    @property
    def validation_names(self) -> list:
        """Create pretty names for the validations that are compared. Should always return 2 values"""
        names = []
        template = "Validation {}:\n{} validated against {}"
        for n, img in enumerate(self._iter_imgs()):
            datasets = img.datasets
            if len(datasets.others)==2:
                for n, ds_meta in enumerate(datasets.others):
                    name = template.format(
                        n, ds_meta["pretty_title"],
                        datasets.ref["pretty_title"]
                    )
                    names.append(name)
                break
            else:
                other = img.datasets.others[0]
                name = template.format(
                    n, other["pretty_title"],
                    img.datasets.ref["pretty_title"]
                )

        return names

    def _check_pairwise(self) -> Union[bool, ComparisonError]:
        """
        Checks that the current initialized supports pairwise comparison methods

        Raise
        -----
        ComparisonException : if not
        """
        pairwise = True
        for n, img in enumerate(self._iter_imgs()):
            if img.datasets.n_datasets() > 2 or n > 1:
                pairwise = False
                break

        if not pairwise:
            raise ComparisonError("For pairwise comparison methods, only two "
                                  "validation results with two datasets each can be compared")

    def get_reference_points(self) -> tuple:
        """
        Get lon, lat arrays for all the reference points in the two validations from the DataArray directly
        (avoid getting them from one of the variables)

        Returns
        -------
        ref_points: np.array
            2D array of lons, lats
        """
        lat, lon = glob.index_names

        lon_list, lat_list = [], []
        for img in self._iter_imgs():
            lon_list.append(img.ds[lon].values)
            lat_list.append(img.ds[lat].values)
        ref_points = np.vstack((
            np.concatenate(lon_list),
            np.concatenate(lat_list),
        )).T

        return ref_points

    def _combine_geometry(
            self,
            imgs:list,
            get_intersection:bool=True,
            return_polys=False
    ) -> tuple:
        """
        Return the union or the intersection of the spatial extents of the provided validations; in case of intersection,
        check that the validations are overlapping

        Parameters
        ----------
        imgs : list
            list with the QA4SMImg corresponding to the paths
        get_intersection : bool, optional. Default is True.
            get extent of the intersection between the two images
        return_polys: bool, default is False.
            whether to return a dictionary with the polygons

        Return
        ------
        extent: tuple
            spatial extent deriving from union or intersection of inputs
        """
        polys = {}

        for n, img in enumerate(imgs):
            minlon, maxlon, minlat, maxlat = img.extent
            bounds = [(minlon,minlat), (maxlon, minlat),
                      (maxlon, maxlat), (minlon, maxlat)]
            Pol = Polygon(bounds)
            name = "Validation {}: ".format(n) + img.name
            polys[name] = Pol

        for n, Pol in enumerate(polys.values()):
            if n == 0:
                # define starting point
                output = Pol
            if not get_intersection or self.single_image:
                # get maximum extent
                output = output.union(Pol)
            # get maximum common
            else:
                output = output.intersection(Pol)
                if not output:
                    raise SpatialExtentError("The spatial extents of the chosen validation results do "
                                             "not overlap. Set 'get_intersection' to False to perform the comparison.")
        polys["selection"] = output

        minlon, minlat, maxlon, maxlat = output.bounds

        if return_polys:
            return (minlon, maxlon, minlat, maxlat), polys

        else:
            return minlon, maxlon, minlat, maxlat

    def visualize_extent(
            self,
            intersection:bool=True,
            plot_points:bool=False,
    ):
        """
        Method to get and visualize the comparison extent including the reference points.

        Parameters
        ----------
        intersection : bool, optional. Default is True.
            choose to visualize the intersection or union output of the comparison
        plot_points : bool, default is False.
            whether to show the reference points in the image
        """
        # self.comparison has not been initialized yet
        imgs = [img for img in self._iter_imgs()]

        extent, polys = self._combine_geometry(
            imgs=imgs,
            get_intersection=intersection,
            return_polys=True,
        )
        ref_points = None
        if plot_points:
            ref_points = self.get_reference_points()

        ref = self._check_ref()["short_name"]
        plot_spatial_extent(
            polys=polys,
            ref_points=ref_points,
            overlapping=self.overlapping,
            intersection_extent=extent,
        )

    def _get_varnames(self, metric):
        """
        Get the list of image Variable names from a metric

        Parameters
        ----------
        metric: str
            name of metric

        Returns
        -------
        varlist: list
            list of the variables
        """
        varlist = []
        for img in self._iter_imgs():
            for Var in img._iter_vars(**{"metric":metric}):
                varlist.append(Var)

        return varlist

    def subset_with_extent(self, dfs:list) -> list:
        """
        Return the original dataframe with only the values included in the selected extent. Basically the
        same method as in QA4SMImg, but it is done here to avoid re-initializing the images

        Returns
        -------
        subset : pd.Series or pd.DataFrame
            initial input with only valid entries in the index
        """
        if self.extent is None:
            return dfs

        lat, lon = glob.index_names
        subset = []
        for df in dfs:
            mask = (df.index.get_level_values(lon) >= self.extent[0]) & (df.index.get_level_values(lon) <= self.extent[1]) &\
                   (df.index.get_level_values(lat) >= self.extent[2]) & (df.index.get_level_values(lat) <= self.extent[3])
            df.where(mask, inplace=True)
            subset.append(df)

        return subset

    def rename_with_stats(self, df):
        """Rename columns of df by adding the content of QA4SMPlotter._box_stats()"""
        renamed = [
            name + QA4SMPlotter._box_stats(df[name]) for name in df.columns
        ]
        df.columns = renamed

        return df

    def _handle_multiindex(self, dfs:list) -> pd.DataFrame:
        """
        Handle ValueError 'cannot handle a non-unique multi-index!' when non-unique multi-index is different in
        the two dfs (e.g. multiple station depths)

        Parameters
        ----------
        dfs : list
            list of (2) dataframes
        """
        try:
            pair_df = pd.concat(dfs, axis=1, join="outer")
        except ValueError:
            pair_df = []
            if self.overlapping:
                # take mean of non-unique values in multi-index (practically speaking, different depths)
                for df in dfs:
                    df = df.groupby(df.index).mean()
                    pair_df.append(df)
            else:
                # take all values; they cannot be compared directly anyway
                for df in dfs:
                    df.reset_index(drop=True, inplace=True)
                    pair_df.append(df)
            pair_df = pd.concat(pair_df, axis=1)

        return pair_df

    def _get_pairwise(self, metric:str) -> pd.DataFrame:
        """
        Get the data and names for pairwise comparisons, meaning: two validations with one satellite dataset each. Includes
        a method to subset the metric values to the selected spatial extent.

        Parameters
        ----------
        metric: str
            name of metric to get data on

        Returns
        -------
        renamed: pd.DataFrame
            Renamed dataframe, ready to be plotted
        """
        to_plot, names = [], []
        # check wether the comparison has one single image and the number of sat datasets
        if self.single_image and not self.perform_checks():
            raise ComparisonError(
                "More than two non-reference datasets are not supported at the moment"
            )

        elif self.single_image:
            for n, Var in enumerate(self._get_varnames(metric)):
                varname = Var.varname
                col_name = "Validation {}:\n{}\n".format(n, Var.pretty_name)
                data = self.comparison._ds2df(varnames=[varname])[varname]
                data = data.rename(col_name)
                to_plot.append(data)

        else:
            for n, (Var, values) in enumerate(
                    zip(self._get_varnames(metric), self.comparison.values())
            ):
                varname = Var.varname
                col_name = "Validation {}:\n{}\n".format(n, Var.pretty_name)
                id, img = values
                data = img._ds2df(varnames=[varname])[varname]
                data = data.rename(col_name)
                to_plot.append(data)

        to_plot = self.subset_with_extent(to_plot)
        pair_df = self._handle_multiindex(to_plot)

        if self.overlapping:
            diff = pair_df.iloc[:,0] - pair_df.iloc[:,1]
            diff = diff.rename(
                "Difference of common points\nbetween validations 0 and 1\n"
            )
            pair_df = pd.concat([pair_df, diff], axis=1)
        renamed = self.rename_with_stats(pair_df)

        return renamed

    def perform_checks(self, overlapping=False, union=False, pairwise=False):
        """Performs selected checks and throws error is they're not passed"""
        if self.single_image:
            return len(self.comparison.datasets.others) <= 2

        # these checks are for multiple images
        else:
            if overlapping:
                if not self.overlapping:
                    raise SpatialExtentError("This method works only in case the initialized "
                                             "validations have overlapping spatial extents.")
            # todo: unexpected behavior here if union is initialized through init_union
            if union and not self.extent:
                if self.union:
                    raise SpatialExtentError("If the comparison is based on the 'union' of spatial extents, "
                                             "this method cannot be called, as it is based on a point-by-point comparison")
            if pairwise:
                self._check_pairwise()

    def diff_table(self, metrics:list) -> pd.DataFrame:  #todo: diff_table for single_image
        """
        Create a table where all the metrics for the different validation runs are compared

        Parameters
        ----------
        metrics: list
            list of metrics to create the table for
        """
        self.perform_checks(pairwise=True)
        table = {}
        for metric in metrics:
            ref = self._check_ref()["short_name"]
            units = glob._metric_description_HTML[metric].format(
                glob._metric_units_HTML[ref]
            )
            description = glob._metric_name[metric] + units
            medians = self._get_pairwise(metric).median()
            # a bit of a hack here
            table[description] = [
                medians[0],
                medians[1],
                medians[0] - medians[1]
            ]
        columns = self.validation_names
        columns.append("Difference of the medians (0 - 1)")
        table = pd.DataFrame.from_dict(
            data=table,
            orient="index",
            columns=columns,
        )

        table = table.applymap(_format_floats)

        return table

    def diff_boxplot(self, metric:str, **kwargs):
        """
        Create a boxplot where two validations are compared. If the comparison is on the subsets union,
        the shown difference corresponds only to the points in the common spatial extent.

        Parameters
        ----------
        metric: str
            metric name which the plot is based on
        """
        self.perform_checks(pairwise=True)
        boxplot_df = self._get_pairwise(metric=metric)
        # prepare axis name
        Metric = QA4SMMetric(metric)
        ref_ds = self.ref['short_name']
        um = glob._metric_description[metric].format(glob._metric_units[ref_ds])
        # plot data
        palette = sns.color_palette(
            palette=['paleturquoise', 'paleturquoise', 'pink'],
            n_colors=3
        )
        fig, axes = boxplot(
            boxplot_df,
            label= "{} {}".format(Metric.pretty_name, um),
            **{'palette':palette}
        )
        # titles for the plot
        fonts = {"fontsize":18}
        title_plot = "Boxplot comparison of {} {}".format(Metric.pretty_name, um)
        axes.set_title(title_plot, pad=glob.title_pad, **fonts)

    def diff_mapplot(self, metric:str, diff_range:str='adjusted', **kwargs):
        """
        Create a pairwise mapplot of the difference between the validations, for a metric. Difference is other - reference

        Parameters
        ----------
        metric: str
            metric from the .nc result file attributes that the plot is based on
        diff_range: str, default is 'adjusted'
            if 'adjusted', colorbar goues from minimum to maximum of difference; if 'fixed', the colorbar goes from the
            maximum to the minimum difference range, by metric
        **kwargs : kwargs
            plotting keyword arguments
        """
        self.perform_checks(overlapping=True, union=True, pairwise=True)
        # todo: fix index names after get_pairwise
        df_diff = self._get_pairwise(metric=metric).dropna()
        Metric = QA4SMMetric(metric)
        um = glob._metric_description[metric].format(glob._metric_units[self.ref['short_name']])
        # make mapplot
        cbar_label = "Difference between {} and {}".format(*df_diff.columns)
        fig, axes = mapplot(
            df_diff.iloc[:,2],  # todo: hack on ids
            metric,
            self.ref['short_name'],
            diff_range=diff_range,
            label=cbar_label
        )
        title_plot = "Overview of the difference in {} {}".format(Metric.pretty_name, um)
        axes.set_title(title_plot, pad=glob.title_pad)

    def wrapper(self, method:str, metric=None, **kwargs):
        """
        Call the method using a list of paths and the already initialised images

        Properties
        ----------
        method: str
            a method from the lookup table in diff_method
        metric: str
            metric from the .nc result file attributes that the plot is based on
        **kwargs : kwargs
            plotting keyword arguments
        """
        diff_methods_lut = {'boxplot': self.diff_boxplot,
                            'mapplot': self.diff_mapplot}
        try:
            diff_method = diff_methods_lut[method]
        except KeyError as e:
            warn(
                'Difference method not valid. Choose one of %s' % ', '.join(diff_methods_lut.keys())
            )
            raise e


        if not metric:
            raise ComparisonError(
                "If you chose '{}' as a method, you should specify"
                " a metric (e.g. 'R').".format(method))

        return diff_method(
            metric=metric,
            **kwargs
        )

im1 = "~/shares/home/Data4projects/qa4sm-reader/Difference_plot_data/0-ISMN.soil moisture_with_1-C3S.sm.middle_US.nc"
im2 = "~/shares/home/Data4projects/qa4sm-reader/Difference_plot_data/0-ISMN.soil moisture_with_1-C3S.sm.west_US.nc"
# im = "~/shares/home/Data4projects/qa4sm-reader/Difference_plot_data/0-ERA5.swvl1_with_1-ESA_CCI_SM_combined.sm_with_2-ESA_CCI_SM_combined.sm.nc"
comp = QA4SMComparison(paths=[im1, im2], get_intersection=True)
