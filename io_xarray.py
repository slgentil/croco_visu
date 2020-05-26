import xarray as xr

# Creation of xarray objects


def open_cdf_dataset(filename, chunks=None, decode_times=True, **kwargs):
    """Return an xarray dataset corresponding to filename.
    Parameters
    ----------
    filename : str
        path to the netcdf file from which to create a xarray dataset
    chunks : dict-like
        dictionnary of sizes of chunk for creating xarray.Dataset.
    Returns
    -------
    ds : xarray.Dataset
    """
    return xr.open_dataset(filename, chunks=chunks, **kwargs)


def open_cdf_mfdataset(filenames, chunks=None, drop_variables=None, **kwargs):
    """Return an xarray dataset corresponding to filename which may include
    wildcards (e.g. file_*.nc).
    Parameters
    ----------
    filename : str
        path to a netcdf file or several netcdf files from which to create a
        xarray dataset
    chunks : dict-like
        dictionnary of sizes of chunk for creating xarray.Dataset.
    Returns
    ------
    ds : xarray.Dataset
    """
    # datasets = []
    # for f in filenames:
    #     try:
    #         ds = xr.open_dataset(f, chunks={'time_counter': 1, 's_rho': 1},\
    #             drop_variables=drop_variables, **kwargs)
    #     except Exception:
    #         ds = xr.open_dataset(f, chunks={'s_rho': 1}, \
    #             drop_variables=drop_variables, **kwargs)
    #     datasets.append(ds)
    # ds = xr.merge(datasets, compat='override')
    # return ds
    return xr.open_mfdataset(filenames, chunks=chunks, drop_variables=drop_variables, **kwargs)


def open_zarr_dataset(path, varnames=None, chunks=None, **kwargs):
    """
    return a xarray dataset corresponding to a zarr archive by variables
    Parameters
    ----------
    - path : str, path to the zarr archive
    - varnames : list,  of the zarr variables to load
    - chunks : dictionnary, chunks of the return dataset
    Return
    ------
    ds : xarray.DataSet
    """

    datasets = []
    for v in varnames:
        ds = xr.open_zarr(path+'%s.zarr'%(v), chunks=chunks, **kwargs)
        datasets.append(ds)
    ds = xr.merge(datasets)

    # On ajoute dans le dataset les paramètres de grille qui sont dans le 1ier fichier
    gridname = path+'../GIGATL6_12h_inst_2004-01-15-2004-01-19.nc'
    gd = xr.open_dataset(gridname, chunks={'s_rho': 1})
    ds['hc'] = gd.hc
    ds['h'] = gd.h
    ds['Vtransform'] = gd.Vtransform
    ds['sc_r'] = gd.sc_r
    ds['sc_w'] = gd.sc_w
    ds['Cs_r'] = gd.Cs_r
    ds['Cs_w'] = gd.Cs_w
    ds['angle'] = gd.angle
    ds['mask_rho'] = gd.mask_rho
    for v in varnames:    
        ds[v] = ds[v].expand_dims({'time_counter':1})
    ds.coords['time'] = xr.DataArray([1.], dims=('time_counter'))
    return ds