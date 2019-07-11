# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.8.0b3 on Tue Jan 30 13:49:27 2018
#

# import sys
import os
# import wx
import numpy as np
# from numpy.matlib import repmat
import numpy.ma as ma
import scipy.io
# import netCDF4 as netcdf
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
# from matplotlib.figure import Figure
from matplotlib import colors
# from matplotlib.widgets  import RectangleSelector


# -------------------------------------------------------------------------
def plotCurv(frame,
             x=None, y=None,
             xlabel=None, ylabel=None,
             legend=None, title=None,
             xlim=None, ylim=None,
             xlog=False, ylog=False):

    # and the axes for the figure

    fig = frame.figure
    fig.clf()
    # axes = fig.add_axes([0.07,0.05,0.98,0.90])
    axes = fig.add_axes([0.08, 0.1, 0.85, 0.85])

    xm = ma.masked_invalid(x)
    ym = ma.masked_invalid(y)

    # axes = self.axes
    if xlim:
        axes.set_xlim((xlim[0], xlim[1]))
    if ylim:
        axes.set_ylim((ylim[0], ylim[1]))

    if y is None:
        if legend is None:
            axes.plot(xm, linewidth=2.0)
        else:
            axes.plot(xm, label=legend, linewidth=2.0)
    elif y.ndim == 1:
        if legend is None:
            axes.plot(xm, ym, linewidth=2.0)
        else:
            axes.plot(xm, ym, label=legend, linewidth=2.0)
    else:
        NbCurv = ym.shape[1]
        for i in range(0, NbCurv):
            if legend is None:
                axes.plot(xm, ym[:, i], linewidth=2.0)
            else:
                axes.plot(xm, ym[:, i], label=legend[i], linewidth=2.0)

    axes.figure.set_facecolor('white')
    axes.grid('on')
    axes.legend()
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if xlog:
        axes.set_xscale('log')
    if ylog:
        axes.set_yscale('log')
    if title is not None:
        axes.set_title(title)


# -------------------------------------------------------------------------
def mypcolor(frame, x, y, z,
             x2=None, xlabel2=None,
             xlim=None, ylim=None,
             clim=None, cformat=None,
             norm=None,
             xlabel=None, ylabel=None, title=None,
             cmap=None, xlog=False, ylog=False,
             z1=None, z2=None, 
             topo=None, nbtopo=None,
             winsize=None, dpi=80):

    zm = ma.masked_invalid(z)
    # # plt.rc('text', usetex=True)

    # # default size if 8 x 6 inch, 80 DPI (640x480 pixels)
    # if winsize is None:
    #     winsize=[8., 6.]
    # fig.set_size_inches( (winsize[0], winsize[1]) )
    fig = frame.figure
    ax = fig.add_axes([0.15, 0.1, 0.85, 0.85])
    ax.callbacks.connect("xlim_changed", frame.notify)
    ax.callbacks.connect("ylim_changed", frame.notify)

    if xlim is None:
        ax.set_xlim((np.min(x), np.max(x)))
    else:
        ax.set_xlim((xlim[0], xlim[1]))
    if ylim is None:
        ax.set_ylim((np.min(y), np.max(y)))
    else:
        ax.set_ylim((ylim[0], ylim[1]))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if cmap is None:
        cmap = DefCmap()

    if clim is not None:
        mesh = ax.pcolormesh(x, y, zm, cmap=cmap, norm=norm, vmin=clim[0], vmax=clim[1])
    else:
        mesh = ax.pcolormesh(x, y, zm, cmap=cmap, norm=norm)

    if z1 is not None:
        # level1 = (z1.min() + z1.max())*0.8
        level1 = (z1.min() + 0) * 0.8
        cp1 = ax.contour(x, y, z1, [level1], colors='b', linewidths=2)
    if z2 is not None:
        # level2 = (z2.min() + z2.max())*0.8
        level2 = (z2.min() + 0) * 0.8
        cp2 = ax.contour(x, y, z2, [level2], colors='r', linewidths=2)
    if topo is not None:
        # Show isocontour of the topography
        cp3 = ax.contour(x, y, topo, nbtopo, colors='grey', linewidths=0.5)
        # Show labels
        # ax.clabel(cp3)

    # Add colorbar
    if cformat == 'sci':
        # make sure to specify tick locations to match desired ticklabels
        cb = fig.colorbar(mesh, ax=ax, format='%.0e', ticks=[clim[0], clim[0] / 10,
                          clim[0] / 100, 0, clim[1] / 100, clim[1] / 10, clim[1]])
        # plt.colorbar(mesh, ax=ax, format='%.0e')
    else:
        cb = fig.colorbar(mesh, ax=ax)

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    if x2 is not None:
        ax2 = ax.twiny()
        mn, mx = ax.get_xlim()
        ax2.set_xlim(2 * np.pi / mn * 1e-3, 2 * np.pi / mx * 1e-3)
        # ax2.set_xlim(2*np.pi/mx, 2*np.pi/mn)
    if xlabel2 is not None:
        ax2.set_xlabel(xlabel2)
    if xlabel2 is not None and xlog:
        ax2.set_xscale('log')

    # font = {'family' : 'normal',
    #         'weight' : 'bold',
    #         'size'   : 18}
    # plt.rc('font', **font)


def DefCmap():

    # get path of launch script croco_gui.py
    script_path = os.path.dirname(os.path.realpath(__file__))
    matfile = scipy.io.loadmat(script_path + '/map_64_wc.mat')
    return array2cmap(np.array(matfile['cm']))


def array2cmap(X):
    N = X.shape[0]

    r = np.linspace(0., 1., N + 1)
    r = np.sort(np.concatenate((r, r)))[1:-1]

    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in range(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in range(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in range(N)])

    rd = tuple([(r[i], rd[i], rd[i]) for i in range(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in range(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in range(2 * N)])

    cdict = {'red': rd, 'green': gr, 'blue': bl}
    return colors.LinearSegmentedColormap('my_colormap', cdict, N)
