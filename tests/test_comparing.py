import threading
import os
import time

import pytest
from qa4sm_reader.comparing import QA4SMComparison, SpatialExtentError

import pandas as pd
import matplotlib.pyplot as plt


# for profiling with cProfile, on the command line run
# python -m cProfile -o ascat_ismn_validation.profile test_validation.py

@pytest.fixture
def single_img():
    testfile = '3-ERA5_LAND.swvl1_with_1-C3S.sm_with_2-ASCAT.sm.nc'
    testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                 'test_data', 'tc', testfile)
    return QA4SMComparison(testfile_path)


@pytest.fixture
def double_img_paths():
    first = '0-ISMN.soil moisture_with_1-C3S.sm.nc'
    second = '0-ISMN.soil moisture_with_1-C3S.sm-overlap.nc'
    testfile_paths = [
        os.path.join(
            os.path.dirname(__file__),
            '..', 'tests', 'test_data', 'comparing',
            i
        ) for i in [
            first,
            second]
    ]
    # initialized with intersection
    return testfile_paths


@pytest.fixture
def double_img_overlap(double_img_paths):
    """Initialized souble image comparison with intersection"""
    return QA4SMComparison(double_img_paths)


def test_init(single_img):
    assert isinstance(single_img.compared, list)


def test_check_ref(single_img):
    assert single_img._check_ref() == {
        'short_name': 'ERA5_LAND',
        'pretty_name': 'ERA5-Land',
        'short_version': 'ERA5_LAND_TEST',
        'pretty_version': 'ERA5-Land test',
        'pretty_title': 'ERA5-Land (ERA5-Land test)'
    }


def test_intersection(double_img_overlap):
    assert not double_img_overlap.union


def test_geometry(double_img_overlap):
    assert double_img_overlap._combine_geometry(double_img_overlap.compared) \
           != double_img_overlap._combine_geometry(double_img_overlap.compared, get_intersection=False)


def test_get_pairwise(single_img, double_img_overlap):
    pair = single_img._get_pairwise("R")

    assert isinstance(pair, pd.DataFrame)
    assert len(pair.columns) == 3, "There should be one column for comparison term" \
                                   "plus the column with difference values"
    pair = double_img_overlap._get_pairwise("R")

    assert isinstance(pair, pd.DataFrame)
    assert len(pair.columns) == 3, "There should be one column for comparison term" \
                                   "plus the column with difference values"


def test_checks(single_img, double_img_overlap):
    """No assertion, but will throw error if any of the checks are not passed"""
    assert single_img.perform_checks()

    double_img_overlap.perform_checks(
        overlapping=True,
        union=True,
        pairwise=True
    )


def test_wrapper(single_img, double_img_overlap):
    """
    This tests the wrapper function but more in general also the
    plotting functions/table
    """
    methods = [
        'boxplot',
        'mapplot'
    ]
    for method in methods:
        out = single_img.wrapper(method, "R")
        plt.close("all")
        assert not out  # generates a plot and returns nothing

    for method in methods:
        out = double_img_overlap.wrapper(method, "R")
        plt.close("all")
        if method == "table":
            assert out is not None  # generates a pandas dataframe
        else:
            assert not out  # generates a plot and returns nothing


def test_init_union(double_img_overlap):
    """Should go at the end as it chnages the attributes"""
    double_img_overlap.init_union()
    assert double_img_overlap.union


def test_pairwise_methods(double_img_paths):
    comp = QA4SMComparison(
        double_img_paths,
        get_intersection=False
    )  # todo: solve unexpected behavior on perform_checks
    works = False
    methods = [
        'boxplot',
        'mapplot'
    ]
    for method in methods:
        try:  # they all have same behavior
            comp.wrapper(
                method,
                metric="R"
            )
        except SpatialExtentError:
            works = True

    assert works


@pytest.fixture
def double_paths_nonoverlap():
    first = '0-ISMN.soil moisture_with_1-C3S.sm.nc'
    second = '0-ISMN.soil moisture_with_1-C3S.sm-nonoverlap.nc'
    testfile_paths = [
        os.path.join(
            os.path.dirname(__file__),
            '..', 'tests', 'test_data', 'comparing',
            i
        ) for i in [
            first,
            second]
    ]
    # initialize the comparison with intersection and check that no error is raised
    QA4SMComparison(
        testfile_paths,
        get_intersection=False
    )

    return testfile_paths


def test_init_error(double_paths_nonoverlap):
    works = False
    try:
        QA4SMComparison(
            double_paths_nonoverlap
        )
    except SpatialExtentError:
        works = True

    assert works


def load_extent_image(paths):
    comp = QA4SMComparison(paths)
    comp.visualize_extent(
        intersection=True,
        plot_points=True
    )
    print("{}".format(time.time()))


def load_table(paths):
    comp = QA4SMComparison(paths)
    metrics = comp.common_metrics
    comp = QA4SMComparison(paths)
    comp.diff_table(metrics=list(metrics.keys()))
    print("{}".format(time.time()))


def load_plots(paths):
    comp = QA4SMComparison(paths)
    metrics = comp.common_metrics
    first_called = list(metrics.keys())[0]
    comp = QA4SMComparison(paths)
    comp.wrapper(
        method="boxplot",
        metric=first_called
    )
    comp = QA4SMComparison(paths)
    comp.wrapper(
        method="mapplot",
        metric=first_called
    )
    print("{}".format(time.time()))


def test_simultaneous_netcdf_loading(double_img_paths):
    # this test should reproduce the calls that are made simultaneously from the server, causing a problem with the
    # netcdf loading function. The calls are made from the view:
    # https://github.com/pstradio/qa4sm/blob/comparison2angular_issue-477/479/api/views/comparison_view.py
    threading.Thread(target=load_extent_image(double_img_paths)).start()
    threading.Thread(target=load_table(double_img_paths)).start()
    threading.Thread(target=load_plots(double_img_paths)).start()
