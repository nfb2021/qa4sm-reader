# -*- coding: utf-8 -*-
import os
import pytest
import tempfile
import shutil

from qa4sm_reader.plotter import QA4SMPlotter
from qa4sm_reader.img import QA4SMImg
from qa4sm_reader.plotting_methods import geotraj_to_geo2d


@pytest.fixture
def plotdir():
    plotdir = tempfile.mkdtemp()

    return plotdir


@pytest.fixture
def basic_plotter(plotdir):
    testfile = '0-ISMN.soil moisture_with_1-C3S.sm.nc'
    testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                      'test_data', 'basic', testfile)
    img = QA4SMImg(testfile_path)
    plotter = QA4SMPlotter(img, plotdir)

    return plotter


@pytest.fixture
def basic_plotter_double(plotdir):
    testfile = '0-GLDAS.SoilMoi0_10cm_inst_with_1-C3S.sm_with_2-SMOS.Soil_Moisture.nc'
    testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                      'test_data', 'basic', testfile)
    img = QA4SMImg(testfile_path)
    plotter = QA4SMPlotter(img, plotdir)

    return plotter


@pytest.fixture
def irrgrid_plotter(plotdir):
    testfile = '0-SMAP.soil_moisture_with_1-C3S.sm.nc'
    testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                 'test_data', 'basic', testfile)
    img = QA4SMImg(testfile_path)
    plotter = QA4SMPlotter(img, plotdir)

    return plotter


@pytest.fixture
def ref_dataset_grid_stepsize(irrgrid_plotter):
    ref_dataset_grid_stepsize = irrgrid_plotter.img.ref_dataset_grid_stepsize

    return ref_dataset_grid_stepsize


@pytest.fixture
def tc_ci_plotter(plotdir):
    testfile = "0-ERA5.swvl1_with_1-ESA_CCI_SM_combined.sm_with_2-ESA_CCI_SM_combined." \
               "sm_with_3-ESA_CCI_SM_combined.sm_with_4-ESA_CCI_SM_combined.sm.CI.nc"
    testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                 'test_data', 'tc', testfile)
    img = QA4SMImg(testfile_path)
    plotter = QA4SMPlotter(img, plotdir)

    return plotter


@pytest.fixture
def tc_plotter(plotdir):
    testfile = '3-GLDAS.SoilMoi0_10cm_inst_with_1-C3S.sm_with_2-SMOS.Soil_Moisture.nc'
    testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                      'test_data', 'tc', testfile)
    img = QA4SMImg(testfile_path)
    plotter = QA4SMPlotter(img, plotdir)

    return plotter


def test_mapplot(basic_plotter, plotdir):
    n_obs_files = basic_plotter.mapplot_metric('n_obs', out_types='png', save_files=True)  # should be 1
    assert len(list(n_obs_files)) == 1
    assert len(os.listdir(plotdir)) == 1

    r_files = basic_plotter.mapplot_metric('R', out_types='svg', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1
    assert len(list(r_files)) == 1

    bias_files = basic_plotter.mapplot_metric('BIAS', out_types='png', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1 + 1
    assert len(list(bias_files)) == 1

    shutil.rmtree(plotdir)  # cleanup


def test_boxplot(basic_plotter, plotdir):
    n_obs_files = basic_plotter.boxplot_basic('n_obs', out_types='png', save_files=True)  # should be 1
    assert len(list(n_obs_files)) == 1
    assert len(os.listdir(plotdir)) == 1

    r_files = basic_plotter.boxplot_basic('R', out_types='svg', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1
    assert len(list(r_files)) == 1

    bias_files = basic_plotter.boxplot_basic('BIAS', out_types='png', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1 + 1
    assert len(list(bias_files)) == 1

    shutil.rmtree(plotdir)


def test_mapplot_double(basic_plotter_double, plotdir):
    n_obs_files = basic_plotter_double.mapplot_metric('n_obs', out_types='png', save_files=True)  # should be 1
    assert len(list(n_obs_files)) == 1
    assert len(os.listdir(plotdir)) == 1

    r_files = basic_plotter_double.mapplot_metric('R', out_types='svg', save_files=True)  # should be 2 files
    assert len(os.listdir(plotdir)) == 1 + 2
    assert len(list(r_files)) == 2

    bias_files = basic_plotter_double.mapplot_metric('BIAS', out_types='png', save_files=True)  # should be 2 files
    assert len(os.listdir(plotdir)) == 1 + 2 + 2
    assert len(list(bias_files)) == 2

    shutil.rmtree(plotdir)


def test_boxplot_double(basic_plotter_double, plotdir):
    n_obs_files = basic_plotter_double.boxplot_basic('n_obs', out_types='png', save_files=True)  # should be 1
    assert len(list(n_obs_files)) == 1
    assert len(os.listdir(plotdir)) == 1

    r_files = basic_plotter_double.boxplot_basic('R', out_types='svg', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1
    assert len(list(r_files)) == 1

    bias_files = basic_plotter_double.boxplot_basic('BIAS', out_types='png', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1 + 1
    assert len(list(bias_files)) == 1

    shutil.rmtree(plotdir)


def test_mapplot_tc(tc_plotter, plotdir):
    n_obs_files = tc_plotter.mapplot_metric('n_obs', out_types='png', save_files=True)  # should be 1
    assert len(list(n_obs_files)) == 1
    assert len(os.listdir(plotdir)) == 1

    r_files = tc_plotter.mapplot_metric('R', out_types='svg', save_files=True)  # should be 2
    assert len(os.listdir(plotdir)) == 1 + 2
    assert len(list(r_files)) == 2

    bias_files = tc_plotter.mapplot_metric('BIAS', out_types='png', save_files=True)  # should be 2
    assert len(os.listdir(plotdir)) == 1 + 2 + 2
    assert len(list(bias_files)) == 2

    snr_files = tc_plotter.mapplot_metric('snr', out_types='svg', save_files=True)  # should be 2
    assert len(os.listdir(plotdir)) == 1 + 2 + 2 + 2
    assert len(list(snr_files)) == 2

    err_files = tc_plotter.mapplot_metric('err_std', out_types='svg', save_files=True)  # should be 2
    assert len(os.listdir(plotdir)) == 1 + 2 + 2 + 2 + 2
    assert len(list(err_files)) == 2

    shutil.rmtree(plotdir)


def test_boxplot_tc(tc_plotter, plotdir):
    n_obs_files = tc_plotter.boxplot_basic('n_obs', out_types='png', save_files=True)  # should be 1
    assert len(list(n_obs_files)) == 1
    assert len(os.listdir(plotdir)) == 1

    r_files = tc_plotter.boxplot_basic('R', out_types='svg', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1
    assert len(list(r_files)) == 1

    bias_files = tc_plotter.boxplot_basic('BIAS', out_types='png', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1 + 1
    assert len(list(bias_files)) == 1

    snr_files = tc_plotter.boxplot_tc('snr', out_types='svg', save_files=True)  # should be 2
    assert len(os.listdir(plotdir)) == 1 + 1 + 1 + 2
    assert len(list(snr_files)) == 2

    err_files = tc_plotter.boxplot_tc('err_std', out_types='svg', save_files=True)  # should be 2
    assert len(os.listdir(plotdir)) == 1 + 1 + 1 + 2 + 2
    assert len(list(err_files)) == 2

    shutil.rmtree(plotdir)


def test_mapplot_irrgrid(irrgrid_plotter, plotdir):
    n_obs_files = irrgrid_plotter.mapplot_metric('n_obs', out_types='png', save_files=True)  # should be 1
    assert len(list(n_obs_files)) == 1
    assert len(os.listdir(plotdir)) == 1

    r_files = irrgrid_plotter.mapplot_metric('R', out_types='svg', save_files=True)  # should be 2
    assert len(os.listdir(plotdir)) == 1 + 1
    assert len(list(r_files)) == 1

    bias_files = irrgrid_plotter.mapplot_metric('BIAS', out_types='png', save_files=True)  # should be 2
    assert len(os.listdir(plotdir)) == 1 + 1 + 1
    assert len(list(bias_files)) == 1

    shutil.rmtree(plotdir)


def test_boxplot_irrgrid(irrgrid_plotter, plotdir):
    n_obs_files = irrgrid_plotter.boxplot_basic('n_obs', out_types='png', save_files=True)  # should be 1
    assert len(list(n_obs_files)) == 1
    assert len(os.listdir(plotdir)) == 1

    r_files = irrgrid_plotter.boxplot_basic('R', out_types='svg', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1
    assert len(list(r_files)) == 1

    bias_files = irrgrid_plotter.boxplot_basic('BIAS', out_types='png', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1 + 1 + 1
    assert len(list(bias_files)) == 1

    shutil.rmtree(plotdir)


def test_grid_creation(irrgrid_plotter, ref_dataset_grid_stepsize):
    metric = 'n_obs'
    for Var in irrgrid_plotter.img._iter_vars(filter_parms={'metric': metric}):
        varname = Var.varname
        df = irrgrid_plotter.img._ds2df([varname])[varname]
        zz, grid, origin = geotraj_to_geo2d(df, grid_stepsize=ref_dataset_grid_stepsize)
        print('varname: ', varname, 'zz: ', zz, 'grid: ', grid)
        assert zz.count() != 0
        assert origin == 'upper'


def test_boxplot_basic_ci(tc_ci_plotter, plotdir):
    bias_files = tc_ci_plotter.boxplot_basic('BIAS', out_types='png', save_files=True)  # should be 1
    assert len(os.listdir(plotdir)) == 1
    assert len(list(bias_files)) == 1

    shutil.rmtree(plotdir)


def test_boxplot_tc_ci(tc_ci_plotter, plotdir):
    snr_files = tc_ci_plotter.boxplot_tc('snr', out_types='svg', save_files=True)  # should be 4
    assert len(os.listdir(plotdir)) == 4
    assert len(list(snr_files)) == 4

    shutil.rmtree(plotdir)


# TODO: make tests for plotting methods
