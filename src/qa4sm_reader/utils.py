from functools import wraps
from typing import Any, Callable, TypeVar, Union
from qa4sm_reader.netcdf_transcription import Pytesmo2Qa4smResultsTranscriber
import xarray as xr
from pathlib import PosixPath

T = TypeVar('T', bound=Callable[..., Any])


def note(note_text: Any) -> Callable[[T], T]:
    """
    Factory function creating a decorator, that prints a note before the execution of the decorated function.

    Parameters:
    ----------
    note_text : Any
        The note to be printed.

    Returns:
    -------
    Callable[[T], T]
        The decorated function.
    """

    def decorator(func: T) -> T:

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f'\n\n{note_text}\n\n')
            return func(*args, **kwargs)

        return wrapper

    return decorator

def transcribe(file_path: Union[str, PosixPath]) ->  Union[None, xr.Dataset]:
    '''If the dataset is not in the new format, transcribe it to the new format.
    This is done under the assumption that the dataset is a `pytesmo` dataset and corresponds to a default\
        validation, i.e. no temporal sub-windows are present.

    Parameters
    ----------
    file_path : str or PosixPath
        path to the file to be transcribed

    Returns
    -------
    dataset : xr.Dataset
        the transcribed dataset
    '''

    temp_sub_wdw_instance = None    # bulk case, no temporal sub-windows

    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=file_path,
        intra_annual_slices=temp_sub_wdw_instance,
        keep_pytesmo_ncfile=False)

    if transcriber.exists:
        return transcriber.get_transcribed_dataset()
