# -*- coding: utf-8 -*-

from qa4sm_reader.plotter import QA4SMPlotter
from qa4sm_reader.img import QA4SMImg
from qa4sm_reader.plot_utils import geotraj_to_geo2d
import os
import unittest
import tempfile
import shutil

class TestQA4SMMetaImgISMNPlotter_newFormat(unittest.TestCase):

    def setUp(self) -> None:
        self.testfile = '0-ISMN.soil moisture_with_1-C3S.sm.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..','tests',
                                          'test_data', 'basic', self.testfile)
        self.plotdir = tempfile.mkdtemp()
        self.img = QA4SMImg(self.testfile_path)
        self.plotter = QA4SMPlotter(self.img, self.plotdir)

    def test_mapplot(self):
        n_obs_files = self.plotter.mapplot('n_obs', out_type='png') # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.mapplot('R', out_type='svg') # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.mapplot('BIAS', out_type='png') # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)

    def test_boxplot(self):
        n_obs_files = self.plotter.boxplot_basic('n_obs', out_type='png') # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.boxplot_basic('R', out_type='svg') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.boxplot_basic('BIAS', out_type='png') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)


class TestQA4SMMetaImgGLDASPlotter_newFormat(unittest.TestCase):

    def setUp(self) -> None:
        self.testfile = '0-GLDAS.SoilMoi0_10cm_inst_with_1-C3S.sm_with_2-SMOS.Soil_Moisture.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..','tests',
                                          'test_data', 'basic', self.testfile)
        self.plotdir = tempfile.mkdtemp()
        self.img = QA4SMImg(self.testfile_path)
        self.plotter = QA4SMPlotter(self.img, self.plotdir)

    def test_mapplot(self):
        n_obs_files = self.plotter.mapplot('n_obs', out_type='png') # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.mapplot('R', out_type='svg') # should be 2 files
        assert len(os.listdir(self.plotdir)) == 1 + 2
        assert len(list(r_files)) == 2

        bias_files = self.plotter.mapplot('BIAS', out_type='png') # should be 2 files
        assert len(os.listdir(self.plotdir)) == 1 + 2 + 2
        assert len(list(bias_files)) == 2

        shutil.rmtree(self.plotdir)

    def test_boxplot(self):
        n_obs_files = self.plotter.boxplot_basic('n_obs', out_type='png') # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.boxplot_basic('R', out_type='svg') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.boxplot_basic('BIAS', out_type='png') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)


class TestQA4SMMetaImgBasicPlotter(unittest.TestCase):

    def setUp(self) -> None:
        self.testfile = '3-GLDAS.SoilMoi0_10cm_inst_with_1-C3S.sm_with_2-SMOS.Soil_Moisture.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..','tests',
                                          'test_data', 'tc', self.testfile)
        self.plotdir = tempfile.mkdtemp()
        self.img = QA4SMImg(self.testfile_path)
        self.plotter = QA4SMPlotter(self.img, self.plotdir)

    def test_mapplot(self):
        n_obs_files = self.plotter.mapplot('n_obs', out_type='png') # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.mapplot('R', out_type='svg') # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 2
        assert len(list(r_files)) == 2

        bias_files = self.plotter.mapplot('BIAS', out_type='png') # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 2 + 2
        assert len(list(bias_files)) == 2


        snr_files = self.plotter.mapplot('snr', out_type='png') # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 2 + 2 + 2
        assert len(list(snr_files)) == 2

        err_files = self.plotter.mapplot('err_std', out_type='svg') # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 2 + 2 + 2 + 2
        assert len(list(err_files)) == 2

        shutil.rmtree(self.plotdir)

    def test_boxplot(self):
        n_obs_files = self.plotter.boxplot_basic('n_obs', out_type='png') # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.boxplot_basic('R', out_type='svg') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.boxplot_basic('BIAS', out_type='png') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        snr_files = self.plotter.boxplot_tc('snr', out_type='png') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1 + 2
        assert len(list(snr_files)) == 2

        err_files = self.plotter.boxplot_tc('err_std', out_type='svg') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1 + 2 + 2
        assert len(list(err_files)) == 2

        shutil.rmtree(self.plotdir)

class TestQA4SMMetaImgIrregularGridPlotter(unittest.TestCase):
    def setUp(self) -> None:
        self.testfile = '0-SMAP.soil_moisture_with_1-C3S.sm.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                          'test_data', 'basic', self.testfile)
        self.plotdir = tempfile.mkdtemp()
        self.img = QA4SMImg(self.testfile_path)
        self.plotter = QA4SMPlotter(self.img, self.plotdir)
        self.ref_dataset_grid_stepsize = self.img.ref_dataset_grid_stepsize

    # def test_netCDF_file:
    #     opened_file =

    def test_mapplot(self):
        n_obs_files = self.plotter.mapplot('n_obs', out_type='png') # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.mapplot('R', out_type='svg') # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.mapplot('BIAS', out_type='png') # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)

    def test_boxplot(self):
        n_obs_files = self.plotter.boxplot_basic('n_obs', out_type='png') # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.boxplot_basic('R', out_type='svg') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.boxplot_basic('BIAS', out_type='png') # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)

    def test_grid_creation(self):
        metric = 'n_obs'
        varnames = list(self.img.metric_meta(metric).keys())
        for varname in varnames:
            df = self.img._ds2df([varname])
            zz, grid, origin = geotraj_to_geo2d(df, varname, grid_stepsize=self.ref_dataset_grid_stepsize)
            print('varname: ', varname, 'zz: ', zz, 'grid: ', grid)
            assert zz.count() != 0
            assert origin == 'upper'

if __name__ == '__main__':
    pass
    # suite = unittest.TestSuite()
    # suite.addTest(TestQA4SMMetaImgGLDASPlotter_newFormat("test_mapplot"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    #
    # suite = unittest.TestSuite()
    # suite.addTest(TestQA4SMMetaImgBasicPlotter("test_boxplot"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)