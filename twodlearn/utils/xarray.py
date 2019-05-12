import typing
import xarray as xr


def _dict_to_xarray(records: dict, dims: list, depth: int):
    '''Transforms a dictionary tree to an xarray recursively.'''
    if depth >= len(dims):
        raise ValueError('Depth of records dictionary is larger than the list '
                         'of dims.')
    if all(isinstance(item, (float, int))
           for item in records.values()):
        series = xr.DataArray(
            data=list(records.values()),
            coords={dims[depth]: list(records.keys())},
            dims=[dims[depth]])
        return series
    array_dict = {
        name: _dict_to_xarray(record, dims, depth+1)
        for name, record in records.items()}
    xarray = xr.concat(
        [r.expand_dims(dims[depth], axis=0)
         for r in array_dict.values()],
        dim=dims[depth]).assign_coords(
        **{dims[depth]: list(array_dict.keys())})
    return xarray


def dict_to_xarray(records: dict,
                   dims: typing.List[str])\
        -> xr.DataArray:
    """Transforms a dictionary tree to an xarray.
    Args:
        records (dict): nested dictionary with the record numbers.
        dims (list): names for the dimensions.
    Returns:
        xr.DataArray: xarray with the dictionary data.
    """
    return _dict_to_xarray(records, dims, 0)
