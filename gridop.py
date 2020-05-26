import sys
import xarray as xr
import numpy as np


from collections import OrderedDict

# ------------------------------------------------------------------------------------------------
_g = 9.81

def _get_spatial_dims(v):
    """ Return an ordered dict of spatial dimensions in the s/z, y, x order
    """
    dims = OrderedDict( (d, next((x for x in v.dims if x[0]==d), None))
                        for d in ['s','y','x'] )
    return dims

def add_coords(ds, var, coords):
    for co in coords:
        var.coords[co] = ds.coords[co]

def rho2u(v, ds):
    """
    interpolate horizontally variable from rho to u point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'lon')
    add_coords(ds, var, ['lon_u','lat_u'])
    var.attrs = v.attrs
    return var.rename(v.name)

def u2rho(v, ds):
    """
    interpolate horizontally variable from u to rho point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'lon')
    add_coords(ds, var, ['lon_r','lat_r'])
    var.attrs = v.attrs
    return var.rename(v.name)

def v2rho(v, ds):
    """
    interpolate horizontally variable from rho to v point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'lat')
    add_coords(ds, var, ['lon_r','lat_r'])
    var.attrs = v.attrs
    return var.rename(v.name)

def rho2v(v, ds):
    """
    interpolate horizontally variable from rho to v point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'lat')
    add_coords(ds, var, ['lon_v','lat_v'])
    var.attrs = v.attrs
    return var.rename(v.name)

def rho2psi(v, ds):
    """
    interpolate horizontally variable from rho to psi point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'lon')
    var = grid.interp(var,'lat')
    var.attrs = v.attrs
    return var.rename(v.name)

def psi2rho(v, ds):
    """
    interpolate horizontally variable from rho to psi point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'lon')
    var = grid.interp(var,'lat')
    add_coords(ds, var, ['lon_r','lat_r'])
    var.attrs = v.attrs
    return var.rename(v.name)


def rho2w(v, ds):
    """
    interpolate horizontally variable from rho to psi point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'s')
    add_coords(ds, var, ['lon_r','lat_r'])
    var.attrs = v.attrs
    return var.rename(v.name)

def w2rho(v, ds):
    """
    interpolate horizontally variable from rho to psi point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'s')
    add_coords(ds, var, ['lon_r','lat_r'])
    var.attrs = v.attrs
    return var.rename(v.name)

def get_hvgrid(ds,varname):
    try:
        var = ds[varname]
    except Exception:
        return 'r', 'r'
    dims = var.dims
    if "x_u" in dims:
        hgrid='u'
    elif "y_v" in dims:
        hgrid='v'
    else:
        hgrid='r'
    if 's_w' in dims:
        vgrid='w'
    else:
        vgrid='r'
    return hgrid, vgrid

def get_z(ds, vgrid='r', hgrid='r', tindex=None):
    z = ds['z_'+vgrid] if tindex is None else ds['z_'+vgrid].isel(t=tindex)
    if hgrid in ['u','v']:
        funtr = eval("rho2"+hgrid)
        z = funtr(z, ds)
    return z

def _get_z(dsr, zeta=None, h=None, vgrid='r', hgrid='r', vtrans=None, tindex=None):
    ''' compute vertical coordinates
        zeta should have the size of the final output
        vertical coordinate is first in output
    '''

    ds = dsr if tindex is None else dsr.isel(t=tindex)
    N = len(ds['s_r'])
    hc = ds.hc

    _h = ds.h if h is None else h
    _zeta = ds.ssh if zeta is None else zeta
    _h = _h.fillna(0.)
    _zeta = _zeta.fillna(0.)

    # swith horizontal grid if needed (should we use grid.interp instead?)
    if hgrid in ['u','v']:
        funtr = eval("rho2"+hgrid)
        _zeta = funtr(_zeta, ds)
        _h = funtr(_h, ds)
    
    # determine what kind of vertical corrdinate we are using (NEW_S_COORD)
    if vtrans is None:
        vtrans = ds.scoord.values
    else:
        if isinstance(vtrans, str):
            if vtrans.lower()=="old":
                vtrans = 1
            elif vtrans.lower()=="new":
                vtrans = 2
            else:
                raise ValueError("unable to understand what is vtransform")
                
    sc=ds['sc_'+vgrid]
    cs=ds['Cs_'+vgrid]

    if vtrans == 2:
        z0 = (hc * sc + _h * cs) / (hc + _h)
        z = _zeta + (_zeta + _h) * z0
        # z = z0 * (_zeta + _h) + _zeta
    else:
        z0 = hc*sc + (_h-hc)*cs
        z = z0 + _zeta*(1+z0/_h)
    
    # reorder spatial dimensions and place them last
    z = z.squeeze(dim=None) 
    sdims = list(_get_spatial_dims(z).values())
    sdims = tuple(filter(None,sdims)) # delete None values
    reordered_dims = tuple(d for d in z.dims if d not in sdims) + sdims
    z = z.transpose(*reordered_dims) 

    return z.rename('z_'+vgrid)


def get_date(ds, tindex=None):
    if tindex is None:
        return ds['time']
    else:
        return ds['time'].values[tindex]

def get_variable(ds, variableName, tindex=None, xindex=None, yindex=None, zindex=None):

        dims = ds[variableName].dims

        variable = ds[variableName]

        if tindex is not None:
            variable = variable.isel(t=tindex)

        if xindex is not None:
            # Extract x slice
            try:
                if "x_u" in dims:
                    variable = variable.isel(x_u=xindex)
                else:
                    variable = variable.isel(x_r=xindex)
            except Exception:
                pass

        if yindex is not None:
            # Extract y slice
            try:
                if "y_v" in dims:
                    variable = variable.isel(y_v=yindex)
                else:
                    variable = variable.isel(y_r=yindex)
            except Exception:
                pass

        if zindex is not None:
            # Extract z slice
            try:
                if "s_w" in dims:
                    variable = variable.isel(s_w=zindex)
                else:
                    variable = variable.isel(s_r=zindex)
            except Exception:
                pass

        return variable

def get_dimname(ds, variableName, axe=None):
    """
    get coordinate corresponding of the variable depending on the direction
    direction : 'x', 'y', 't'
    """
    # If variable in the dataset
    try:
        dims = ds[variableName].dims
        dim = [s for s in dims if axe.replace('z','s')+'_' in s]
        return dim[0] if dim else None
    except Exception:
        if axe in ['x','y']:
            return axe+'_r'
        else:
            if variableName in ['zeta_k']:
                return axe.replace('z','s')+'_r'
            else:
                return axe.replace('z','s')+'_w'


def get_coord(ds, variable, axe=None):
    """
    get coordinate corresponding of the variable depending on the direction
    direction : 'x', 'y', 't'
    """
    # If variable is derived variable, return rho point coordinates
    coords = variable.coords
    if axe=='x':
        coordname = [s for s in coords if 'lon_' in s][0]
    elif axe == 'y':
        coordname = [s for s in coords if 'lat_' in s][0]
    elif axe == 'z':
        coordname = [s for s in coords if 'z' in s][0]
    else:
        print('get_coord: unknown axis')
        return None
    return variable[coordname]


def find_nearest_above(my_array, target, axis=0):
    diff = target - my_array
    diff = diff.where(diff>0,np.inf)
    return xr.DataArray(diff.argmin(axis=axis))


def section(ds, var, z, longitude=None, latitude=None, depth=None):
        """
        #
        #
        # This function interpolate a 3D variable on a slice at a constant depth or
        # constant longitude or constant latitude
        #
        # On Input:
        #
        #    ds      dataset to find the grid
        #    var     (dataArray) Variable to process (3D matrix).
        #    z       (dataArray) Depths at the same point than var (3D matrix).
        #    longitude   (scalar) longitude of the slice (scalar meters, negative).
        #    latitude    (scalar) latitude of the slice (scalar meters, negative).
        #    depth       (scalar) depth of the slice (scalar meters, negative).
        #
        # On Output:
        #
        #    vnew    (dataArray) Horizontal slice
        #
        #
        """

        # if z.compute().shape != var.compute().shape:
        if z.shape != var.shape:
            print('slice: var and z shapes are different')
            return

        # Find dimensions of the variable
        xdim = [s for s in var.dims if "x_" in s][0]
        ydim = [s for s in var.dims if "y_" in s][0]
        zdim = [s for s in var.dims if "s_" in s][0]

        N = len(var[zdim])

        # Find horizontal coordinates of the variable
        x = [var.coords[s] for s in var.coords if "lon_" in s][0]
        y = [var.coords[s] for s in var.coords if "lat_" in s][0]
        s = [var.coords[s] for s in var.coords if "s_" in s][0]

        # Find the indices of the grid points just below the longitude/latitude/depth
        if longitude is not None:
            axe = x.get_axis_num(xdim)
            indices = find_nearest_above(x, longitude, axis=axe)
        elif latitude is not None:
            axe = y.get_axis_num(ydim)
            indices = find_nearest_above(y, latitude, axis=axe)
        elif depth is not None:
            axe = z.get_axis_num(zdim)
            indices = find_nearest_above(z, depth, axis=axe)
        else:
            "Longitude or latitude or depth must be defined"
            return None

        # Initializes the 2 slices around the longitude/latitude/depth
        if longitude is not None:
            x1 = x.isel({xdim:indices})
            x2 = x.isel({xdim:indices+1})
            y1 = y.isel({xdim:indices})
            y2 = y.isel({xdim:indices+1})
            z1 = z.isel({xdim:indices})
            z2 = z.isel({xdim:indices+1})
            v1 = var.isel({xdim:indices})
            v2 = var.isel({xdim:indices+1})
        elif latitude is not None:
            x1 = x.isel({ydim:indices})
            x2 = x.isel({ydim:indices+1})
            y1 = y.isel({ydim:indices})
            y2 = y.isel({ydim:indices+1})
            z1 = z.isel({ydim:indices})
            z2 = z.isel({ydim:indices+1})
            v1 = var.isel({ydim:indices})
            v2 = var.isel({ydim:indices+1})
        elif depth is not None:
            z1 = z.isel({zdim:indices})
            z2 = z.isel({zdim:indices+1})
            v1 = var.isel({zdim:indices})
            v2 = var.isel({zdim:indices+1})

        # Do the linear interpolation
        if longitude is not None:
            xdiff = x1 - x2
            ynew =  (((y1 - y2) * longitude + y2 * x1 - y1 * x2) / xdiff)
            znew =  (((z1 - z2) * longitude + z2 * x1 - z1 * x2) / xdiff)
            vnew =  (((v1 - v2) * longitude + v2 * x1 - v1 * x2) / xdiff)
        elif latitude is not None:
            ydiff = y1 - y2
            xnew =  (((x1 - x2) * latitude + x2 * y1 - x1 * y2) / ydiff)
            znew =  (((z1 - z2) * latitude + z2 * y1 - z1 * y2) / ydiff)
            vnew =  (((v1 - v2) * latitude + v2 * y1 - v1 * y2) / ydiff)
        elif depth is not None:
            zmask = z1 * 0. + 1
            zmask = zmask.where(z1<depth,np.nan)
            vnew =  zmask * (((v1 - v2) * depth + v2 * z1 - v1 * z2) / (z1 - z2))

        # Add the coordinates to dataArray
        if longitude is not None:
            ynew = ynew.expand_dims({s.name: N})
            vnew = vnew.assign_coords(coords={"z":znew})
            vnew = vnew.assign_coords(coords={y.name:ynew})

        elif latitude is not None:
            xnew = xnew.expand_dims({s.name: N})
            vnew = vnew.assign_coords(coords={"z":znew})
            vnew = vnew.assign_coords(coords={x.name:xnew})

        elif depth is not None:
            vnew = vnew.assign_coords(coords={y.name:y})
            vnew = vnew.assign_coords(coords={x.name:x})

        return vnew

def rotuv(ds, hgrid='r'):

    '''
    Rotate winds or u,v to lat,lon coord -> result on rho grid by default
    '''

    angle = ds.angle
    u=u2rho(ds.u,ds)
    v=v2rho(ds.v,ds)

    cosang = np.cos(angle)
    sinang = np.sin(angle)
    urot = u*cosang - v*sinang
    vrot = u*sinang + v*cosang
    

    if hgrid in ['u','v']:
        funtr = eval("rho2"+hgrid)
        urot = funtr(urot, ds)
        vrot = funtr(vrot, ds)

    return [urot,vrot]


def get_p(grid,rho,zw,zr=None,g=_g):
    """ compute (not reduced) pressure by integration from the surface, 
    taking rho at rho points and giving results on w points (z grid)
    with p=0 at the surface. If zr is not None, compute result on rho points """
    if zr is None:
        dz = grid.diff(zw, "s")
        p = grid.cumsum((rho*dz).sortby("s_rho",ascending=False), "s",                             to="outer", boundary="fill").sortby("s_w",ascending=False)
    else:
        """ it is assumed that all fields are from bottom to surface"""
        rna = {"s_w":"s_rho"}
        dpup = (zr - zw.isel(s_w=slice(0,-1)).drop("s_w").rename(rna))*rho
        dpdn = (zw.isel(s_w=slice(1,None)).drop("s_w").rename(rna) - zr)*rho
        p = (dpup.shift(s_rho=-1, fill_value=0) + dpdn).sortby(rho.s_rho, ascending=False)                .cumsum("s_rho").sortby(rho.s_rho, ascending=True).assign_coords(z_r=zr)
    return _g *p.rename("p")


def get_uv_from_psi(psi, ds):
    # note that u, v are computed at rho points
    x, y = ds.xi_rho, ds.eta_rho
    #
    u = - 0.5*(psi.shift(eta_rho=1)-psi)/(y.shift(eta_rho=1)-y) \
        - 0.5*(psi-psi.shift(eta_rho=-1))/(y-y.shift(eta_rho=-1))
    #
    v =   0.5*(psi.shift(xi_rho=1)-psi)/(x.shift(xi_rho=1)-x) \
        + 0.5*(psi-psi.shift(xi_rho=-1))/(x-x.shift(xi_rho=-1))
    return u, v


def interp2z_3d(z0, z, v, extrap):
    import crocosi.fast_interp3D as fi  # OpenMP accelerated C based interpolator
    # check v and z have identical shape
    assert v.ndim==z.ndim
    # test if temporal dimension is present
    if v.ndim == 1:
        lv = v.squeeze()[:,None,None]
        lz = z.squeeze()[:,None,None]
    elif v.ndim == 2:
        lv = v[...,None]
        lz = z[...,None]
    else:
        lz = z[...]
        lv = v[...]
    #
    if extrap:
        zmin = np.min(z0)-1.
        lv = np.concatenate((lv[[0],...], lv), axis=0)
        lz = np.concatenate((zmin+0.*lz[[0],...], lz), axis=0)
    #
    return fi.interp(z0.astype('float64'), lz.astype('float64'), lv.astype('float64'))

def interp2z(z0, z, v, extrap):
    ''' interpolate vertically
    '''
    # check v and z have identical shape
    assert v.ndim==z.ndim
    # test if temporal dimension is present
    if v.ndim == 4:
        vi = [interp2z_3d(z0, z[...,t], v[...,t], extrap)[...,None] for t in range(v.shape[-1])]
        return np.concatenate(vi, axis=0) # (50, 722, 258, 1)
        #return v*0 + v.shape[3]
    else:
        return interp2z_3d(z0, z, v, extrap)


def N2Profile(run, strat, z, g=9.81):
    """
    Method to compute the N2 profile : 
    """
    
    grid = run.ds['his'].attrs['xgcm-Grid']
    N2 = -g/run.params_input['rho0'] * grid.diff(strat,'s') / grid.diff(z,'s')
    N2.isel(s_w=0).values = N2.isel(s_w=1).values
    N2.isel(s_w=-1).values = N2.isel(s_w=-2).values
    # if np.any(N2<0):
    #     print("Unstable N2 profile detected")
    return (N2)

def hinterp(ds,var,coords=None):
    import pyinterp
    #create Tree object
    mesh = pyinterp.RTree()

    L = ds.dims['x_r']
    M = ds.dims['y_r']
    N = ds.dims['s_r']
    z_r = get_z(ds)
    #coords = np.array([coords])

    # where I know the values
    z_r = get_z(ds)
    vslice = []
    #lon_r = np.tile(ds.lon_r.values,ds.dims['s_r']).reshape(ds.dims['s_r'], ds.dims['y_r'], ds.dims['x_r'])
    lon_r = np.tile(ds.lon_r.values,(ds.dims['s_r'],1,1)).reshape(ds.dims['s_r'], ds.dims['y_r'], ds.dims['x_r'])
    lat_r = np.tile(ds.lat_r.values,(ds.dims['s_r'],1,1)).reshape(ds.dims['s_r'], ds.dims['y_r'], ds.dims['x_r'])
    z_r = get_z(ds)
    mesh.packing(np.vstack((lon_r.flatten(), lat_r.flatten(), z_r.values.flatten())).T,
                            var.values.flatten())

    # where I want the values
    zcoord = np.zeros_like(ds.lon_r.values)
    zcoord[:] = coords
    vslice, neighbors = mesh.inverse_distance_weighting(
        np.vstack((ds.lon_r.values.flatten(), ds.lat_r.values.flatten(), zcoord.flatten())).T,
        within=True)    # Extrapolation is forbidden)


    # The undefined values must be set to nan.
    print(ds.mask_rho.values)
    mask=np.where(ds.mask_rho.values==1.,)
    vslice[int(ds.mask_rho.values)] = float("nan")

    vslice = xr.DataArray(np.asarray(vslice.reshape(ds.dims['y_r'], ds.dims['x_r'])),dims=('x_r','y_r'))
    yslice = ds.lat_r
    xslice = ds.lon_r

    return[xslice,yslice,vslice]
