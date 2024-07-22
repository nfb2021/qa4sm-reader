import os
import pytest
from copy import deepcopy
from datetime import datetime
import xarray as xr
import shutil
from pathlib import Path

from qa4sm_reader.netcdf_transcription import Pytesmo2Qa4smResultsTranscriber
from qa4sm_reader.intra_annual_temp_windows import TemporalSubWindowsCreator, NewSubWindow
import tempfile
import qa4sm_reader.globals as qr_globals


@pytest.fixture
def bulk_case_file(TEST_DATA_DIR: Path) -> Path:
    testfile =  Path('basic', '6-ISMN.soil moisture_with_1-C3S.sm_with_2-C3S.sm_with_3-SMOS.Soil_Moisture_with_4-SMAP.soil_moisture_with_5-ASCAT.sm.nc')

    src = TEST_DATA_DIR / testfile
    assert src.exists()

    tmpdir = Path(tempfile.mkdtemp()) / testfile.parent
    tmpdir.mkdir(parents=True, exist_ok=True)

    dst = tmpdir / testfile.name

    shutil.copy(src, dst)
    return dst

#------------------Check that all required consts are defined------------------
def test_qr_globals_attributes():
    attributes = [
        'METRICS', 'TC_METRICS', 'NON_METRICS', 'METADATA_TEMPLATE',
        'IMPLEMENTED_COMPRESSIONS', 'ALLOWED_COMPRESSION_LEVELS',
        'INTRA_ANNUAL_METRIC_TEMPLATE', 'INTRA_ANNUAL_TCOL_METRIC_TEMPLATE',
        'TEMPORAL_SUB_WINDOW_SEPARATOR', 'DEFAULT_TSW',
        'TEMPORAL_SUB_WINDOW_NC_COORD_NAME', 'MAX_NUM_DS_PER_VAL_RUN',
        'DATASETS', 'TEMPORAL_SUB_WINDOWS'
    ]

    assert any(attr in dir(qr_globals) for attr in attributes)

    assert 'zlib' in qr_globals.IMPLEMENTED_COMPRESSIONS

    assert qr_globals.ALLOWED_COMPRESSION_LEVELS == [None, *list(range(10))]

    assert qr_globals.INTRA_ANNUAL_METRIC_TEMPLATE == [
        "{tsw}", qr_globals.TEMPORAL_SUB_WINDOW_SEPARATOR, "{metric}"
    ]

    assert qr_globals.INTRA_ANNUAL_TCOL_METRIC_TEMPLATE == qr_globals.INTRA_ANNUAL_TCOL_METRIC_TEMPLATE == [
        "{tsw}", qr_globals.TEMPORAL_SUB_WINDOW_SEPARATOR, "{metric}", "_",
        "{number}-{dataset}", "_between_"
    ]

    assert len(qr_globals.TEMPORAL_SUB_WINDOW_SEPARATOR) == 1

    assert qr_globals.TEMPORAL_SUB_WINDOWS == {
        "seasons": {
            "S1": [[12, 1], [2, 28]],
            "S2": [[3, 1], [5, 31]],
            "S3": [[6, 1], [8, 31]],
            "S4": [[9, 1], [11, 30]],
        },
        "months": {
            "Jan": [[1, 1], [1, 31]],
            "Feb": [[2, 1], [2, 28]],
            "Mar": [[3, 1], [3, 31]],
            "Apr": [[4, 1], [4, 30]],
            'May': [[5, 1], [5, 31]],
            "Jun": [[6, 1], [6, 30]],
            "Jul": [[7, 1], [7, 31]],
            "Aug": [[8, 1], [8, 31]],
            "Sep": [[9, 1], [9, 30]],
            "Oct": [[10, 1], [10, 31]],
            "Nov": [[11, 1], [11, 30]],
            "Dec": [[12, 1], [12, 31]],
        }
    }


# ------------------Test Pytesmo2Qa4smResultsTranscriber-------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

#-------------------Test instantiation of Pytesmo2Qa4smResultsTranscriber and basic functionalities------------

def test_on_non_existing_file():
    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results='non_existing.nc',
        intra_annual_slices=None,
        keep_pytesmo_ncfile=False
    )

    assert not transcriber.exists

def test_keep_pytesmo_ncfile(bulk_case_file):
    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=str(bulk_case_file),
        intra_annual_slices=None,
        keep_pytesmo_ncfile=True)

    assert transcriber.exists


    transcriber.output_file_name = bulk_case_file
    transcribed_ds = transcriber.get_transcribed_dataset()

    assert isinstance(transcribed_ds, xr.Dataset)

    transcriber.write_to_netcdf(transcriber.output_file_name)

    assert transcriber.output_file_name.exists()

    assert Path(bulk_case_file.parent, bulk_case_file.name + '.old').exists()

    os.remove(Path(bulk_case_file.parent, bulk_case_file.name + '.old'))





#-------------------Test default case (= no temporal sub-windows)----------------------------------------------

def test_bulk_case_transcription(bulk_case_file):
    ...


if __name__ == '__main__':
    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results='non_existing.nc',
        intra_annual_slices=None,
        keep_pytesmo_ncfile=True
    )

print(os.path.join(
        os.path.dirname(__file__), 'test_data', 'basic',
        '6-ISMN.soil moisture_with_1-C3S.sm_with_2-C3S.sm_with_3-SMOS.Soil_Moisture_with_4-SMAP.soil_moisture_with_5-ASCAT.sm.nc'
    ))
