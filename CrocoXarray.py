# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.8.0b3 on Tue Jan 30 13:49:27 2018
#

import os
# import sys
import re
# import wx
import numpy as np
import xarray as xr
# import netCDF4 as netcdf
from croco_wrapper import CrocoWrapper
# from io_xarray import return_xarray_dataarray, return_xarray_dataset

second2day = 1. /86400.

class Croco(object):
    '''
    Croco class grouping all the methods relative to the variables
    '''
    def __init__(self):
        '''
        Initialise the Croco object
        '''

        self.wrapper = CrocoWrapper()

        self.r_earth = 6371315. # Mean earth radius in metres (from scalars.h)
        
        # An index along either x or y dimension
        self.ij = None

        self.ListOfVariables = self.list_of_variables()
        self.ListOfDerived = self.list_of_derived()

        self.rho0 = 1025.
        self.g = 9.81

    def list_of_variables(self):
        '''
        return names of variables depending on time and having at least 3 dimensions
        '''
        # open file
        # ds = return_xarray_dataset(self.crocofile)
        # retrieve real name for time dimension
        # timedim = self.wrapper.keymap_dimensions['tdim']
        ds = self.wrapper.dsvar
        keys = []
        self.variables = {}
        # retrieve variable name
        for key in ds.data_vars.keys():
            if ('t' in ds[key].dims and len(ds[key].dims)>2):
                self.variables[key] = ds[key]
                keys.append(key)
        return keys

    def list_of_derived(self):
        ''' List of calculated variables implemented '''
        keys = []
        keys.append('pv_ijk')
        keys.append('zeta_k')
        keys.append('dtdz')
        return keys

    def get_variable(self,variableName, tindex=None, xindex=None, yindex=None, zindex=None):

        list = self.variables[variableName].dims

        variable =  self.variables[variableName]

        if tindex is not None:
            variable = variable.isel(t=tindex)

        if xindex is not None:
            # find x dimension of the variable
            regex=re.compile(".*(x_).*")
            dim = [m.group(0) for l in list for m in [regex.search(l)] if m]
            # Extract x slice
            try:
                if dim[0].find('u') >=0 :
                    variable = variable.isel(x_u=xindex)
                elif dim[0].find('v') >=0 :
                    variable = variable.isel(x_v=xindex)
                elif dim[0].find('r') >=0 :
                    variable = variable.isel(x_r=xindex)
                elif dim[0].find('w') >=0 :
                    variable = variable.isel(x_w=xindex)
            except: 
                pass

        if yindex is not None:
            # find y dimension of the variable
            regex=re.compile(".*(y_).*")
            dim = [m.group(0) for l in list for m in [regex.search(l)] if m]
            # Extract y slice
            try:
                if dim[0].find('u') >=0 :
                    variable = variable.isel(y_u=yindex)
                elif dim[0].find('v') >=0 :
                    variable = variable.isel(y_v=yindex)
                elif dim[0].find('r') >=0 :
                    variable = variable.isel(y_r=yindex)
                elif dim[0].find('w') >=0 :
                    variable = variable.isel(y_w=yindex)
            except: 
                pass

        if zindex is not None:
            # find z dimension of the variable
            regex=re.compile(".*(z_).*")
            dim = [m.group(0) for l in list for m in [regex.search(l)] if m]
            # Extract y slice
            try:
                if dim[0].find('r') >=0 :
                    variable = variable.isel(z_r=zindex)
                elif dim[0].find('w') >=0 :
                    variable = variable.isel(z_w=zindex)
            except: 
                pass

        return variable


    def get_coord(self,variableName, direction=None, timeIndex=None):
        """
        get coordinate corresponding of the variable depending on the direction
        direction : 'x', 'y', 't'
        """
        # If variable is derived variable, return rho point coordinates
        try:
            list = self.variables[variableName].dims
        except:
            try:
                list = self.variables['rho'].dims
            except:
                list = self.variables['temp'].dims
        if direction == 'x':
            regex=re.compile(".*(x_).*")
            coord = [m.group(0) for l in list for m in [regex.search(l)] if m]
            if coord[0].find('u') >=0 :
                return  self.wrapper.coords['lon_u']
            elif coord[0].find('v') >=0 :
                return  self.wrapper.coords['lon_v']
            elif coord[0].find('r') >=0 :
                return  self.wrapper.coords['lon_r']
            elif coord[0].find('w') >=0 :
                return  self.wrapper.coords['lon_w']
        elif direction == 'y':
            regex=re.compile(".*(y_).*")
            coord = [m.group(0) for l in list for m in [regex.search(l)] if m]
            if coord[0].find('u') >=0 :
                return  self.wrapper.coords['lat_u']
            elif coord[0].find('v') >=0 :
                return  self.wrapper.coords['lat_v']
            elif coord[0].find('r') >=0 :
                return  self.wrapper.coords['lat_r']
            elif coord[0].find('w') >=0 :
                return  self.wrapper.coords['lat_w']
        elif direction == 'z':    
            ssh = self.variables['ssh'].isel(t=timeIndex).values
            regex=re.compile(".*(z_).*")
            coord = [m.group(0) for l in list for m in [regex.search(l)] if m]
            if coord[0].find('r') >=0 :                
            	z = self.wrapper.scoord2z_r(ssh, alpha=0., beta=0)
            elif coord[0].find('w') >=0 :
            	z = self.wrapper.scoord2z_w(ssh, alpha=0., beta=0)
            if variableName=="u":
                z = self.rho2u_3d(z)
            elif variableName=="v":
                z = self.rho2v_3d(z)
            return(z)
        elif direction == 't':
            return  self.wrapper.coords['time'].values

    def create_DataArray(self, data=None, dimstyp='xyzt'):
        ''' 
        Create a xarrayr DataArray avec les dimensions possible x,y,z et t.
        Par defaut la variable est au point rho.
        '''

        # Create dims
        dims=[]
        # dims = ('t', 'z_r', 'y_r', 'x_r')
        if 't' in dimstyp: dims.append('t')
        if 'z' in dimstyp: dims.append('z_r')
        if 'y' in dimstyp: dims.append('y_r')
        if 'x' in dimstyp: dims.append('x_r')

        var = xr.DataArray(data=data, dims=dims)
        return var


    def zslice(self,var,mask,z,depth,findlev=False):
        """
        #
        #
        # This function interpolate a 3D variable on a horizontal level of constant
        # depth
        #
        # On Input:  
        # 
        #    var     Variable to process (3D matrix).
        #    z       Depths (m) of RHO- or W-points (3D matrix).
        #    depth   Slice depth (scalar meters, negative).
        # 
        # On Output: 
        #
        #    vnew    Horizontal slice (2D matrix). 
        #
        #
        """
        [N,Mp,Lp]=z.shape

        #
        # Find the grid position of the nearest vertical levels
        #
        a=np.where(z<=depth,1,0)
        levs=np.squeeze(np.sum(a,axis=0)) - 1
        levs = np.where(levs==N-1,N-2,levs)
        levs = np.where(levs==-1,0,levs)
        minlev = np.min(levs)
        maxlev = np.max(levs)
        if findlev==True:
            return minlev,maxlev

        # Do the interpolation
        z1 = np.zeros_like(z[0,:,:])
        z2 = np.zeros_like(z[0,:,:])
        v1 = np.zeros_like(z[0,:,:])
        v2 = np.zeros_like(z[0,:,:])

        for j in range(Mp):
            for i in range(Lp):
                k = levs[j,i]
                z1[j,i] = z[k+1,j,i]
                z2[j,i] = z[k,j,i]
                v1[j,i] = var[k+1,j,i]
                v2[j,i] = var[k,j,i]
        zmask = np.where(z2>depth,np.nan,1)
        vnew=mask*zmask*(((v1-v2)*depth+v2*z1-v1*z2)/(z1-z2))
        return vnew,minlev,maxlev


    def get_run_name(self):
        filename = self.wrapper.keymap_files['variable_file']
        index = filename.find("/CROCO_FILES")
        if index == -1:
            runName = ''
        else:
            runName = os.path.basename(os.path.dirname(filename[:index]))
        return runName

    @staticmethod
    def rho2u_2d(rho_in):
        '''
        Convert a 2D field at rho points to a field at u points
        '''
        def _r2u(rho_in, Lp):
            u_out = rho_in[:, :Lp - 1]
            u_out += rho_in[:, 1:Lp]
            u_out *= 0.5
            return u_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        Mshp, Lshp = rho_in.shape
        return _r2u(rho_in, Lshp)

    @staticmethod
    def rho2u_3d(rho_in):
        '''
        Convert a 3D field at rho points to a field at u points
        Calls rho2u_2d
        '''
        def levloop(rho_in):
            Nlevs, Mshp, Lshp = rho_in.shape
            rho_out = np.zeros((Nlevs, Mshp, Lshp-1))
            for k in xrange(Nlevs):
                 rho_out[k] = Croco.rho2u_2d(rho_in[k])
            return rho_out
        assert rho_in.ndim == 3, 'rho_in must be 3d'
        return levloop(rho_in)

    @staticmethod
    def rho2v_2d(rho_in):
        '''
        Convert a 2D field at rho points to a field at v points
        '''
        def _r2v(rho_in, Mp):
            v_out = rho_in[:Mp - 1]
            v_out += rho_in[1:Mp]
            v_out *= 0.5
            return v_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        Mshp, Lshp = rho_in.shape
        return _r2v(rho_in, Mshp)

    @staticmethod
    def rho2v_3d(rho_in):
        '''
        Convert a 3D field at rho points to a field at v points
        Calls rho2v_2d
        '''
        def levloop(rho_in):
            Nlevs, Mshp, Lshp = rho_in.shape
            rho_out = np.zeros((Nlevs, Mshp-1, Lshp))
            for k in xrange(Nlevs):
                 rho_out[k] = Croco.rho2v_2d(rho_in[k])
            return rho_out
        assert rho_in.ndim == 3, 'rho_in must be 3d'
        return levloop(rho_in)


    @staticmethod
    def u2rho_2d(u_in):
        '''
        Convert a 2D field at u points to a field at rho points
        '''
        def _uu2ur(uu_in, Mp, Lp):
            L, Lm = Lp - 1, Lp - 2
            u_out = np.zeros((Mp, Lp))
            u_out[:, 1:L] = 0.5 * (u_in[:, 0:Lm] + \
                                   u_in[:, 1:L])
            u_out[:, 0] = u_out[:, 1]
            u_out[:, L] = u_out[:, Lm]
            return u_out.squeeze()

        assert u_in.ndim == 2, 'u_in must be 2d'
        Mp, Lp = u_in.shape
        return _uu2ur(u_in, Mp, Lp+1)

    @staticmethod
    def u2rho_3d(u_in):
        '''
        Convert a 3D field at u points to a field at rho points
        Calls u2rho_2d
        '''
        def _levloop(u_in):
            Nlevs, Mshp, Lshp = u_in.shape
            u_out = np.zeros((Nlevs, Mshp, Lshp+1))
            for Nlev in xrange(Nlevs):
                u_out[Nlev] = Croco.u2rho_2d(u_in[Nlev])
            return u_out
        assert u_in.ndim == 3, 'u_in must be 3d'
        return _levloop(u_in)

    @staticmethod
    def v2rho_2d(v_in):
        '''
        Convert a 2D field at v points to a field at rho points
        '''
        def _vv2vr(v_in, Mp, Lp):
            M, Mm = Mp - 1, Mp - 2
            v_out = np.zeros((Mp, Lp))
            v_out[1:M] = 0.5 * (v_in[:Mm] + \
                                   v_in[1:M])
            v_out[0] = v_out[1]
            v_out[M] = v_out[Mm]
            return v_out.squeeze()

        assert v_in.ndim == 2, 'v_in must be 2d'
        Mp, Lp = v_in.shape
        return _vv2vr(v_in, Mp+1, Lp)

    @staticmethod
    def v2rho_3d(v_in):
        '''
        Convert a 3D field at v points to a field at rho points
        Calls v2rho_2d
        '''
        def levloop(v_in):
            Nlevs, Mshp, Lshp = v_in.shape
            v_out = np.zeros((Nlevs, Mshp+1, Lshp))
            for Nlev in xrange(Nlevs):
                v_out[Nlev] = Croco.v2rho_2d(v_in[Nlev])
            return v_out
        assert v_in.ndim == 3, 'v_in must be 3d'
        return levloop(v_in)
