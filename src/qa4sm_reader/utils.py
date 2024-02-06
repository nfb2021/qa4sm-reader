import xarray as xr

def make_loc_dim(ds: xr.Dataset) -> xr.Dataset:
    """
    Takes a dataset with coords 'idx', 'lon, 'lat' (possibly amongst others) and combines them into a single 'loc' dimension, the new main dimension of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be processed.

    Returns
    -------
    xr.Dataset
        Dataset with 'loc' as the main dimension.
    """

    eds_idx_data = ds['idx'].data
    eds_idx_attrs = ds['idx'].attrs
    eds_lon_attrs = ds['lon'].attrs
    eds_lat_attrs = ds['lat'].attrs

    eds = ds.drop_vars(['idx', 'lon', 'lat'])
    eds['idx'] = eds_idx_data
    eds['idx'].attrs = eds_idx_attrs
    eds = eds.swap_dims({'idx': 'loc'})
    eds['lon'] = eds['longitude']
    eds['lon'].attrs = eds_lon_attrs
    eds['lat'] = eds['latitude']
    eds['lat'].attrs = eds_lat_attrs
    eds = eds.set_coords(['lon', 'lat', 'idx'])
    return eds
