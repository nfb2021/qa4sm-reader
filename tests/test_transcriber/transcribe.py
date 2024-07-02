from transcriber.intra_annual_temp_windows import TemporalSubWindowsCreator, NewSubWindow
from transcriber.netcdf_transcription import Pytesmo2Qa4smResultsTranscriber

from pathlib import Path, PosixPath
import os
from glob import glob
import shutil

import numpy as np
import xarray as xr
from typing import List, Dict, Union, Tuple, Optional
# all testdata is exclusively of the bulk case
# initialize the corresponding temporal sub-windows
temp_sub_wdw_instance, temp_sub_wdws = None, None



class TestDataTranscriber:
    def __init__(self,
                 test_data_pth: Optional[PosixPath] = None,
                 temporal_sub_windows: Optional[List[str]]=None,
                 transcribed_nc_dir_name: Optional[str]=None,
                 *args, **kwargs):
        if not test_data_pth:
            test_data_pth = Path.cwd() / 'tests' / 'test_data'
        self.test_data_pth = test_data_pth

        self.temp_sub_wdw_instance = temporal_sub_windows

        if not transcribed_nc_dir_name:
            transcribed_nc_dir_name = 'test_qa4sm_data'
        self.transcribed_nc_dir_name = transcribed_nc_dir_name

    @property
    def old_test_nc_pths(self) -> List[PosixPath]:
        return [x for x in Path.cwd().joinpath(self.test_data_pth).glob('**/*.nc')]


    def copy_nc_files(self) -> List[PosixPath]:
        common_prefix = self.test_data_pth
        restructured_files_root = common_prefix.parent / self.transcribed_nc_dir_name
        if restructured_files_root.exists():
            shutil.rmtree(restructured_files_root)

        nc_files = list(common_prefix.glob('**/**/*.nc'))
        unique_dirs = {file.parent for file in nc_files}

        for dir_path in unique_dirs:
            relative_path = dir_path.relative_to(common_prefix)
            new_dir_path = restructured_files_root / relative_path
            new_dir_path.mkdir(parents=True, exist_ok=True)

            for nc_file in dir_path.glob('*.nc'):
                shutil.copy(nc_file, new_dir_path)
                # pass

        print(f"All .nc files have been copied to '{restructured_files_root}'.")
        return list(restructured_files_root.glob('**/*.nc'))

    def encoding_params(self, ds: xr.Dataset, compression: str, complevel: int) -> Dict:
        encoding = {}
        for var in ds.variables:
            if not np.issubdtype(ds[var].dtype, np.object_):
                encoding[var] = {compression: True, 'complevel': complevel, 'shuffle': True}
            else:
                encoding[var] = {'zlib': False}
        return encoding

    def transcribe(self, file_path: PosixPath) ->  None:
        _transcriber = Pytesmo2Qa4smResultsTranscriber(
            pytesmo_results=file_path,
            intra_annual_slices=self.temp_sub_wdw_instance,
            keep_pytesmo_ncfile=False)

        if _transcriber.exists:
            restructured_results = _transcriber.get_transcribed_dataset()
            encoding=self.encoding_params(restructured_results, 'zlib', 4)
            out_name = file_path.with_name(f"{file_path.stem}.qa4sm")
            restructured_results.to_netcdf(out_name, encoding=encoding)
            os.remove(file_path)
            os.rename(out_name, file_path)
        return None


if __name__ == '__main__':

    transcriber = TestDataTranscriber()

    to_be_transcribed = transcriber.copy_nc_files()
    unsuccesfull = []
    for f in to_be_transcribed:
        transcriber.transcribe(f)
