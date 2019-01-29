# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.8.0b3 on Tue Jan 30 13:49:27 2018
#

# import os
# import sys
# import re
# import wx
import numpy as np
# import xarray as xr
# import netCDF4 as netcdf
# from croco_wrapper import CrocoWrapper
# from io_xarray import return_xarray_dataarray, return_xarray_dataset

       
###############################################################
# Ertel Potential vorticity


def get_pv(croco,tindex,depth=None, minlev=None, maxlev=None, \
	       lonindex=None, latindex=None, typ='ijk'):

    mask = croco.wrapper.masks['mask_r']
    pv = np.full_like(mask,np.nan)

    # pv from minlev to maxlev
    if depth is None or depth<=0:
        pv = np.tile(pv,(maxlev-minlev,1,1))
        pv[:,1:-1,1:-1] = calc_ertel(croco,tindex,minlev=minlev,maxlev=maxlev,typ=typ)
    # pv on a level
    elif depth > 0:
        pv[1:-1,1:-1] = calc_ertel(croco,tindex,minlev=int(depth)-2,maxlev=int(depth-1),typ=typ) 
    return pv


def calc_ertel(croco,tindex, minlev=None, maxlev=None, typ='ijk'):                  
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #   epv    - The ertel potential vorticity with respect to property 'lambda'
    #
    #                                       [ curl(u) + f ]
    #   -  epv is given by:           EPV = --------------- . del(lambda)
    #                                            rho
    #
    #   -  pvi,pvj,pvk - the x, y, and z components of the potential vorticity.
    #
    #   -  Ertel PV is calculated on horizontal rho-points, vertical w-points.
    #
    #
    #   tindex   - The time index at which to calculate the potential vorticity.
    #   depth    - depth 
    #
    # Adapted from rob hetland.
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #
    # Grid parameters
    #
    pm = croco.wrapper.metrics['dx_r']
    pm = np.tile(pm,(maxlev-minlev+1,1,1))
    pn = croco.wrapper.metrics['dy_r']
    pn = np.tile(pn,(maxlev-minlev+1,1,1))
    f = croco.wrapper.metrics['f']
    f = np.tile(f,(maxlev-minlev+1,1,1))
    rho0=croco.rho0
    #
    # 3D variables
    #
    ssh = croco.variables['ssh'].isel(t=tindex).values
    dz = croco.wrapper.scoord2dz_r(ssh, alpha=0., beta=0)[minlev:maxlev+1,:,:]

    u = croco.variables['u'].isel(t=tindex, z_r=slice(minlev,maxlev+1))
    v = croco.variables['v'].isel(t=tindex, z_r=slice(minlev,maxlev+1))
    w = croco.variables['w'].isel(t=tindex, z_r=slice(minlev,maxlev+1))

    try:
        rho = croco.variables['rho'].isel(t=tindex, z_r=slice(minlev,maxlev+1))
    except:
        # temp = croco.variables['temp'].isel(t=tindex, z_r=slice(minlev,maxlev+1))
        # salt = croco.variables['salt'].isel(t=tindex, z_r=slice(minlev,maxlev+1))
        # rho = croco.rho_eos(temp,salt,0)
        print('rho not in history file')
        return
    
    if 'k' in typ:
        #
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #  Ertel potential vorticity, term 1: [f + (dv/dx - du/dy)]*drho/dz
        #
        # Compute d(v)/d(xi) at PSI-points.
        #
        dxm1 = 0.25 * (pm[:,:-1,1:]+pm[:,1:,1:]+pm[:,:-1,:-1]+pm[:,1:,:-1])
        dvdxi = np.diff(v ,n=1,axis=2) * dxm1
        #
        #  Compute d(u)/d(eta) at PSI-points.
        #
        dym1 = 0.25 * (pn[:,:-1,1:]+pn[:,1:,1:]+pn[:,:-1,:-1]+pn[:,1:,:-1])
        dudeta = np.diff(u ,n=1,axis=1) * dym1
        #
        #  Compute d(rho)/d(z) at horizontal RHO-points and vertical W-points
        #
        dz_w = 0.5*(dz[:-1,:,:]+dz[1:,:,:])
        drhodz = np.diff(rho,n=1,axis=0) / dz_w
        #
        #  Compute Ertel potential vorticity <k hat> at horizontal RHO-points and
        #  vertical W-points.
        omega = dvdxi - dudeta
        omega = f[:,1:-1,1:-1] + 0.25 * (omega[:,:-1,1:]+omega[:,1:,1:]+omega[:,:-1,:-1]+omega[:,1:,:-1])
        pvk = 0.5*(omega[:-1,:,:]+omega[1:,:,:]) * drhodz[:,1:-1,1:-1]
    else:
        pvk = 0.

    if 'i' in typ:
        #
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #  Ertel potential vorticity, term 2: (dw/dy - dv/dz)*(drho/dx)
        #
        #  Compute d(w)/d(y) at horizontal V-points and vertical RHO-points
        #
        dym1 = 0.5 * (pn[:,:-1,:]+pn[:,1:,:])
        dwdy = np.diff(w,axis=1)*dym1
        #
        #  Compute d(v)/d(z) at horizontal V-points and vertical W-points
        #
        dz_v = 0.5 * (dz[:,1:,:]+dz[:,:-1,:])
        # dz_v = croco.crocoGrid.scoord2dz_v(zeta, alpha=0., beta=0.)
        dvdz = np.diff(v,axis=0)/(0.5*(dz_v[:-1,:,:]+dz_v[1:,:,:]))
        #
        #  Compute d(rho)/d(xi) at horizontal U-points and vertical RHO-points
        #
        dxm1 = 0.5 * (pm[:,:,1:]+pm[:,:,:-1])
        drhodx = np.diff(rho,axis=2) * dxm1
        #
        #  Add in term 2 contribution to Ertel potential vorticity at horizontal RHO-points and
        #  vertical W-points.
        #
        pvi = ( 0.25*(dwdy[1:,:-1,1:-1]+dwdy[1:,1:,1:-1]+
        	          dwdy[:-1,:-1,1:-1]+dwdy[:-1,1:,1:-1]) - \
        	  0.5*(dvdz[:,:-1,1:-1]+dvdz[:,1:,1:-1])) * \
              0.25*(drhodx[1:,1:-1,:-1]+drhodx[1:,1:-1,1:]+ \
              	    drhodx[:-1,1:-1,:-1]+drhodx[:-1,1:-1,1:])
    else:
        pvi = 0.

    if 'j' in typ:
        #
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #  Ertel potential vorticity, term 3: (du/dz - dw/dx)*(drho/dy)
        #
        #  Compute d(u)/d(z) at horizontal U-points and vertical W-points
        #
        dz_u = 0.5 * (dz[:,:,1:]+dz[:,:,:-1])
        dudz = np.diff(u,axis=0)/(0.5*(dz_u[:-1,:,:]+dz_u[1:,:,:]))
        #
        #  Compute d(w)/d(x) at horizontal U-points and vertical RHO-points
        #
        dxm1 = 0.5 * (pm[:,:,1:]+pm[:,:,:-1])
        dwdx = np.diff(w,axis=2)*dxm1
        #
        #  Compute d(rho)/d(eta) at horizontal V-points and vertical RHO-points
        #
        dym1 = 0.5 * (pn[:,1:,:]+pn[:,:-1,:])
        drhodeta = np.diff(rho,axis=1) * dym1
        #
        #  Add in term 3 contribution to Ertel potential vorticity at horizontal RHO-points and
        #  vertical W-points..
        #
        pvj = ( 0.5*(dudz[:,1:-1,1:]+dudz[:,1:-1,:-1]) - \
                0.25*(dwdx[1:,1:-1,1:]+dwdx[1:,1:-1,:-1]+ \
                	dwdx[:-1,1:-1,1:]+dwdx[:-1,1:-1,:-1]))* \
              0.25*(drhodeta[1:,:-1,1:-1]+drhodeta[1:,1:,1:-1]+ \
              	    drhodeta[:-1,:-1,1:-1]+drhodeta[:-1,1:,1:-1])
    else:
        pvj = 0.

    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Sum potential vorticity components, and divide by rho0
    #
    pvi =  pvi/rho0
    pvj =  pvj/rho0
    pvk =  pvk/rho0
    #
    return(pvi + pvj + pvk)
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ####################################################################################


###############################################################
# Zeta_k term

def get_zetak(croco,tindex,depth=None, minlev=None, maxlev=None):

    mask = croco.wrapper.masks['mask_r']
    pv = np.full_like(mask,np.nan)

    # pv from minlev to maxlev
    if depth is None or depth<=0:
        pv = np.tile(pv,(maxlev-minlev,1,1))
        pv[:,1:-1,1:-1] = calc_zetak(croco,tindex,minlev=minlev,maxlev=maxlev-1)
    # pv on a level
    elif depth > 0:
        pv[1:-1,1:-1] = calc_zetak(croco,tindex,minlev=int(depth)-1,maxlev=int(depth-1)) 
    return pv


def calc_zetak(croco,tindex, minlev=None, maxlev=None):                  
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #   
    #
    #                                      
    #   -  zetak is given by:      (dv/dx - du/dy)/f  
    #
    #   -  zetak is calculated at RHO-points
    #
    #
    #   tindex   - The time index at which to calculate the potential vorticity.
    #   depth    - depth 
    #
    # Adapted from rob hetland.
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #
    # Grid parameters
    #
    pm = croco.wrapper.metrics['dx_r']
    pm = np.tile(pm,(maxlev-minlev+1,1,1))
    pn = croco.wrapper.metrics['dy_r']
    pn = np.tile(pn,(maxlev-minlev+1,1,1))
    f = croco.wrapper.metrics['f']
    f = np.tile(f,(maxlev-minlev+1,1,1))
    #
    # 3D variables
    #

    u = croco.variables['u'].isel(t=tindex, z_r=slice(minlev,maxlev+1))
    v = croco.variables['v'].isel(t=tindex, z_r=slice(minlev,maxlev+1))
    
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #  Ertel potential vorticity, term 1: (dv/dx - du/dy)/f
    #
    # Compute d(v)/d(xi) at PSI-points.
    #
    dxm1 = 0.25 * (pm[:,:-1,1:]+pm[:,1:,1:]+pm[:,:-1,:-1]+pm[:,1:,:-1])
    dvdxi = np.diff(v ,n=1,axis=2) * dxm1
    #
    #  Compute d(u)/d(eta) at PSI-points.
    #
    dym1 = 0.25 * (pn[:,:-1,1:]+pn[:,1:,1:]+pn[:,:-1,:-1]+pn[:,1:,:-1])
    dudeta = np.diff(u ,n=1,axis=1) * dym1
    #
    #  Compute Ertel potential vorticity <k hat> at horizontal RHO-points and
    #  vertical RHO-points.
    omega = dvdxi - dudeta
    return( 0.25 * (omega[:,:-1,1:]+omega[:,1:,1:]+ \
    	             omega[:,:-1,:-1]+omega[:,1:,:-1]) / f[:,1:-1,1:-1] )
    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ####################################################################################



###############################################################
# dtdz term

def get_dtdz(croco,tindex,depth=None, minlev=None, maxlev=None):

    mask = croco.wrapper.masks['mask_r']
    dtdz = np.full_like(mask,np.nan)

    # dtdz from minlev to maxlev
    if depth is None or depth<=0:
        dtdz = np.tile(dtdz,(maxlev-minlev,1,1))
        dtdz[:] = calc_dtdz(croco,tindex,minlev=minlev,maxlev=maxlev)
    # dtdz on a level
    elif depth > 0:
        dtdz[:] = calc_dtdz(croco,tindex,minlev=int(depth)-2,maxlev=int(depth)-1) 
    return dtdz

def calc_dtdz(croco,tindex, minlev=None, maxlev=None):

    #
    # 3D variables
    #
    ssh = croco.variables['ssh'].isel(t=tindex).values
    dz = croco.wrapper.scoord2dz_r(ssh, alpha=0., beta=0)[minlev:maxlev+1,:,:]
    t = croco.variables['temp'].isel(t=tindex, z_r=slice(minlev,maxlev+1))
    dtdz = np.diff(t,axis=0) / (0.5*(dz[:-1,:,:]+dz[1:,:,:]))
    return(dtdz)
