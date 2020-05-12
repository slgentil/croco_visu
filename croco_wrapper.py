# -*- coding: UTF-8 -*-
#
import sys
import numpy as np
import xarray as xr
from xgcm import Grid
from io_xarray import *
from gridop import *
from gridop import _get_z

second2day = 1. / 86400.

path = "./"
# path = "/Users/slgentil/models/croco/gigatl/giga_2004a2014_mean/../"


# Either a string of the form "dummy*.nc" 
# or an explicit list of files to open ['dummy1.nc', 'dummy2.nc']
files = 'moz_his.nc'
# files = 'GIGATL6_12h_inst_2004-01-15-2004-01-19.nc'
# varnames = ['zeta', 'u', 'v', 'rho']

keymap_dimensions = {
    'x_rho': 'x_r',
    'x_u': 'x_u',
    'y_rho': 'y_r',
    'y_v': 'y_v',
    's_rho': 's_r',
    's_w': 's_w',
    'time_counter': 't'
}


keymap_coordinates = {
    'nav_lon_rho': 'lon_r',
    'nav_lat_rho': 'lat_r',
    'nav_lon_u': 'lon_u',
    'nav_lat_u': 'lat_u',
    'nav_lon_v': 'lon_v',
    'nav_lat_v': 'lat_v',
    'nav_lon_w': 'lon_w',
    'nav_lat_w': 'lat_w',
    'nav_lon_psi': 'lon_p',
    'nav_lat_psi': 'lat_p',
    'time_instant': 'time'
}


keymap_variables = {
    'ssh': 'ssh',
    'u': 'u',
    'v': 'v',
    'w': 'w',
    'temp': 'temp',
    'salt': 'salt',
    'rho': 'rho'
}

keymap_metrics = {
    'pm': 'dx_r',
    'pn': 'dy_r',
    'theta_s': 'theta_s',
    'theta_b': 'theta_b',
    'Vtransform': 'scoord',
    'hc': 'hc',
    'h': 'h',
    'f': 'f'
}

keymap_masks = {
    'mask_rho': 'mask_r'
}

# Variables holders for croco


class CrocoWrapper(object):
    """This class create the dictionnary of variables used for creating a
    generic grid from xios croco output files.
    """
    def __init__(self, chunks=None, mask_level=0):

        # self.keymap_files = keymap_files
        self.keymap_dimensions = keymap_dimensions
        self.keymap_coordinates = keymap_coordinates
        self.keymap_variables = keymap_variables
        self.keymap_metrics = keymap_metrics
        self.keymap_masks = keymap_masks

        self.openfiles()
        self.change_dimensions()
        self.change_metrics()
        self.change_coordinates()
        self.change_masks()
        self.change_variables()
        self.parameters = {}
        #self.parameters['chunks'] = chunks


    def __getitem__(self, key):
            """ Load data set by providing suffix
            """
            # assert key in self.open_nc
            if key in ["dscoord","dsmetrics","dsmask","dsvar"]:
                return self.key

    def _get(self, *args, **kwargs):
        return return_xarray_dataarray(*args, **kwargs)

    def _get_date(self, tindex):
        return self.ds['time'].values[tindex]

    def openfiles(self):
        # self.ds = open_cdf_mfdataset(path+files, chunks={'time_counter': 1, 's_rho':1}).load()
        self.ds = open_cdf_mfdataset(path+files, chunks={'time_counter': 1},\
            drop_variables=['lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v',\
            'ubar','vbar','sustr','svstr','bvf','temp','salt','w']).load()
        # self.ds = open_zarr_dataset(path, varnames=varnames,\
            # chunks={'time_counter': 1, 's_rho':1}).load()

    def change_dimensions(self):
        ds = self.ds
        for key, val in self.keymap_dimensions.items():
            try:
                ds = ds.rename({key: val})
            except Exception:
                pass

        # rename redundant dimensions
        _dims = (d for d in ['x_v', 'y_u', 'x_w', 'y_w'] if d in ds.dims)
        for d in _dims:
            ds = ds.rename({d: d[0]+'_r'})

        # Create xgcm grid
        coords={'lon':{'center':'x_r', 'inner':'x_u'}, 
                'lat':{'center':'y_r', 'inner':'y_v'}, 
                's':{'center':'s_r', 'outer':'s_w'}}
        ds.attrs['xgcm-Grid'] = Grid(ds, coords=coords)

        self.ds = ds

    def change_coordinates(self):
        #ds = return_xarray_dataset(self.keymap_files['coordinate_file'])
        ds = self.ds
        #ds = self.change_coords(ds)
        for key, val in self.keymap_coordinates.items():
            try:
                ds = ds.rename({key: val})
            except Exception:
                pass

        # change nav variables to coordinates        
        _coords = [d for d in [d for d in ds.data_vars.keys()] if "nav_" in d]
        ds = ds.set_coords(_coords) 
        # change name of new coords
        for c in ds.coords:
            new_c = c.replace('nav_lat','lat').replace('nav_lon','lon')
            ds = ds.rename({c:new_c})
        
        # rename coordinates 
        eta_suff={}
        for c in ds.coords:
            new_c = c.replace('nav_lat','lat').replace('nav_lon','lon')
            ds = ds.rename({c:new_c}) 

        # remove coordinates *_1point
        _coords = [d for d in [d for d in ds.coords.keys()] if "_1point" in d]
        ds = ds.drop(_coords)

        # Compute  coordinates if zero
        ds.coords['lon_u'] = rho2u(ds.lon_r, ds)
        ds.coords['lat_u'] = rho2u(ds.lat_r, ds)
        ds.coords['lon_v'] = rho2v(ds.lon_r, ds)
        ds.coords['lat_v'] = rho2v(ds.lat_r, ds)

        # Add vertical coordinates
        ds['z_r'] = _get_z(ds).persist()
        ds['z_w'] = _get_z(ds, vgrid='w').persist()

        # # time = time - time_origin
        ds['time'].values = np.array(ds['time'].values, dtype='datetime64[s]') - \
            np.array(ds['time'].time_origin, dtype='datetime64[s]')
        ds['time'].values = ds['time'].values / np.timedelta64(1, 'D')

        self.ds = ds

    def change_metrics(self):
        ds = self.ds
        for key, val in self.keymap_metrics.items():
            try:
                ds = ds.rename({key: val})
            except Exception:
                pass
        self.ds = ds

    def change_masks(self):
        ds = self.ds
        for key, val in self.keymap_masks.items():
            try:
                ds = ds.rename({key: val})
            except Exception:
                pass
        if "mask_r" not in ds.data_vars:
            mask_r = ds.lon_r * 0. + 1.
            ds["mask_r"] = mask_r
        self.ds = ds

    def change_variables(self):
        ds = self.ds
        for key, val in self.keymap_variables.items():
            try:
                ds = ds.rename({key: val})
            except Exception:
                pass
        self.ds = ds

    def chunk(self, chunks=None):
        """
        Chunk all the variables.
        Parameters
        ----------
        chunks : dict-like
            dictionnary of sizes of chunk along xarray dimensions.
        """
        for dataname in self.variables:
            data = self.variables[dataname]
            if isinstance(data, xr.DataArray):
                self.variables[dataname] = data.chunk(chunks)



# Run the program

if __name__ == '__main__':
    run = CrocoWrapper()
    ds = run.ds.isel(t=0)
    grid = ds.attrs['xgcm-Grid']
    pm = ds['dx_r']
    u = ds['u']
    v = ds['v']
    pmpsi = rho2psi(pm,ds)
    vdiff = v.diff('x_r')
    vdiff = grid.diff(v, axis="lon")
    print(pmpsi)
    print(vdiff)
