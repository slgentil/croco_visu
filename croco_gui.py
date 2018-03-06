# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.8.0b3 on Tue Jan 30 13:49:27 2018
#

import sys
import os
import wx
import numpy as np
from numpy.matlib import repmat
import numpy.ma as ma
import scipy.io
import netCDF4 as netcdf
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colors
from Croco import Croco
from myplot import plotCurv, mypcolor

wildcard = "Netcdf Files (*.nc)|*.nc"
# figsize = [10,9]
figsize = [6,5]

# begin wxGlade: dependencies
# end wxGlade

# begin wxGlade: extracode
# end wxGlade

########################################################################

class SectionFrame(wx.Frame):

    def __init__(self):

        """Constructor"""

        wx.Frame.__init__(self, None, wx.ID_ANY, title='Section')

        self.panel = wx.Panel(self, wx.ID_ANY)

        self.figure = Figure()
        self.axes = self.figure.add_axes([0,0,1,1])
        self.canvas = FigureCanvas(self.panel, -1, self.figure)

        self.AnimationBtn = wx.Button(self.panel, wx.ID_ANY, "Animation")
        self.AnimationBtn.Bind(wx.EVT_BUTTON, self.onAnimationBtn)
        self.startTimeTxt = wx.TextCtrl(self.panel, wx.ID_ANY, "1", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.startTimeTxt.Bind(wx.EVT_TEXT_ENTER, self.onstartTimeTxt)
        self.endTimeTxt = wx.TextCtrl(self.panel, wx.ID_ANY, "1", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.endTimeTxt.Bind(wx.EVT_TEXT_ENTER, self.onendTimeTxt)
        self.ZoomInBtn = wx.Button(self.panel, wx.ID_ANY, "Zoom In")
        self.ZoomInBtn.Bind(wx.EVT_BUTTON, self.onZoomInBtn)
        self.ZoomOutBtn = wx.Button(self.panel, wx.ID_ANY, "Zoom Out")
        self.ZoomOutBtn.Bind(wx.EVT_BUTTON, self.onZoomOutBtn)
        self.PrintBtn = wx.Button(self.panel, wx.ID_ANY, "Print")
        self.PrintBtn.Bind(wx.EVT_BUTTON, self.onPrintBtn)
        
        self.ResetColorBtn = wx.Button(self.panel, wx.ID_ANY, "Reset Color")
        self.ResetColorBtn.Bind(wx.EVT_BUTTON, self.onResetColorBtn)
        self.MinColorTxt = wx.TextCtrl(self.panel, wx.ID_ANY, "Min Color", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.MinColorTxt.Bind(wx.EVT_TEXT_ENTER, self.onMinColorTxt)
        self.MaxColorTxt = wx.TextCtrl(self.panel, wx.ID_ANY, "Max Color", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.MaxColorTxt.Bind(wx.EVT_TEXT_ENTER, self.onMaxColorTxt)

        self.__do_layout()

    def __do_layout(self):

        topSizer        = wx.BoxSizer(wx.VERTICAL)
        canvasSizer     = wx.BoxSizer(wx.VERTICAL)
        buttonsSizer    = wx.BoxSizer(wx.HORIZONTAL)
        colorSizer      = wx.BoxSizer(wx.HORIZONTAL)


        canvasSizer.Add(self.canvas, 0, wx.ALL, 5)
        buttonsSizer.Add(self.AnimationBtn,0, wx.ALL, 5)
        buttonsSizer.Add(self.startTimeTxt,1, wx.ALL, 5)
        buttonsSizer.Add(self.endTimeTxt,1, wx.ALL, 5)
        buttonsSizer.Add(self.ZoomInBtn,0, wx.ALL, 5)
        buttonsSizer.Add(self.ZoomOutBtn,0, wx.ALL, 5)
        buttonsSizer.Add(self.PrintBtn,0, wx.ALL, 5)
        colorSizer.Add(self.ResetColorBtn, 0, wx.ALL, 5)
        colorSizer.Add(self.MinColorTxt, 0, wx.ALL, 5)
        colorSizer.Add(self.MaxColorTxt, 0, wx.ALL, 5)

        topSizer.Add(canvasSizer, 0, wx.CENTER)
        topSizer.Add(buttonsSizer, 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(colorSizer, 0, wx.ALL|wx.EXPAND, 5)

        self.panel.SetSizer(topSizer)
        topSizer.Fit(self)

        self.Layout()

    def onFigureClick(self,event):
        self.xPress, self.yPress = event.xdata, event.ydata

    def onFigureRelease(self,event):
        self.xRelease, self.yRelease = event.xdata, event.ydata
        # self.lonReleaseIndex,self.latReleaseIndex = self.findLatLonIndex(self.lonRelease, self.latRelease)

    def onAnimationBtn(self,event):
        print("Animation")

    def onstartTimeTxt(self,event):
        print("startTimeTxt")

    def onendTimeTxt(self,event):
        print("endTimeTxt")

    def onZoomInBtn(self,event):
        self.xlim = [min(self.xPress,self.xRelease),max(self.xPress,self.xRelease)]
        self.ylim = [min(self.yPress,self.yRelease),max(self.yPress,self.yRelease)]
        self.drawz(setlim=False)

    def onZoomOutBtn(self,event):
        self.xlim = [np.min(self.x),np.max(self.x)]
        self.ylim = [np.min(self.y),np.max(self.y)]
        self.drawz()

    def onPrintBtn(self,event):
        filename = self.variableName + ".png"
        self.figure.savefig(filename, dpi=self.figure.dpi)

    def onResetColorBtn(self,event):
        self.clim = [np.min(self.variableZ),np.max(self.variableZ)]
        self.MinColorTxt.SetValue('%.2E' % self.clim[0])
        self.MaxColorTxt.SetValue('%.2E' % self.clim[1])
        self.drawz(setlim=False)

    def onMinColorTxt(self,event):
        self.clim[0] = float(self.MinColorTxt.GetValue())
        self.drawz(setlim=False)

    def onMaxColorTxt(self,event):
        self.clim[1] = float(self.MaxColorTxt.GetValue())
        self.drawz(setlim=False)

    def drawz(self, setlim=True):
        self.canvas.Destroy()
        self.figure = Figure(figsize=(figsize[0],figsize[1]))
        self.canvas = FigureCanvas(self.panel, -1, self.figure)
        self.canvas.mpl_connect('button_press_event', self.onFigureClick)
        self.canvas.mpl_connect('button_release_event', self.onFigureRelease)

        if setlim:
            self.clim = [np.min(self.variableZ),np.max(self.variableZ)]
            self.mincolor = np.min(self.variableZ)
            self.MinColorTxt.SetValue('%.2E' % self.mincolor)
            self.maxcolor = np.max(self.variableZ)
            self.MaxColorTxt.SetValue('%.2E' % self.maxcolor)
            self.xlim = [np.min(self.x),np.max(self.x)]
            self.ylim = [np.min(self.y),np.max(self.y)]
        title = "{:s}, {:s}={:4.1f}, Time={:4.1f}".format(self.variableName,self.section,self.latlon,self.time)
        mypcolor(self.figure,self.x,self.y,self.variableZ,\
                      title=title,\
                      xlabel=self.section,\
                      ylabel='Depth',\
                      xlim=self.xlim,\
                      ylim=self.ylim,\
                      clim=self.clim)

        self.canvas.draw()
        self.canvas.Refresh()
        self.Show()


########################################################################

class ProfileFrame(wx.Frame):

    def __init__(self):

        """Constructor"""

        wx.Frame.__init__(self, None, wx.ID_ANY, title='Profile')

        self.panel = wx.Panel(self, wx.ID_ANY)

        self.figure = Figure()
        # self.axes = self.figure.add_axes([0.1,0.1,0.9,0.9])
        self.canvas = FigureCanvas(self.panel, -1, self.figure)

        self.ZoomInBtn = wx.Button(self.panel, wx.ID_ANY, "Zoom In")
        self.ZoomInBtn.Bind(wx.EVT_BUTTON, self.onZoomInBtn)
        self.ZoomOutBtn = wx.Button(self.panel, wx.ID_ANY, "Zoom Out")
        self.ZoomOutBtn.Bind(wx.EVT_BUTTON, self.onZoomOutBtn)
        self.PrintBtn = wx.Button(self.panel, wx.ID_ANY, "Print")
        self.PrintBtn.Bind(wx.EVT_BUTTON, self.onPrintBtn)


        self.__do_layout()

    def __do_layout(self):

        topSizer        = wx.BoxSizer(wx.VERTICAL)
        canvasSizer     = wx.BoxSizer(wx.VERTICAL)
        buttonsSizer    = wx.BoxSizer(wx.HORIZONTAL)


        canvasSizer.Add(self.canvas, 0, wx.ALL, 5)
        buttonsSizer.Add(self.ZoomInBtn,0, wx.ALL, 5)
        buttonsSizer.Add(self.ZoomOutBtn,0, wx.ALL, 5)
        buttonsSizer.Add(self.PrintBtn,0, wx.ALL, 5)

        topSizer.Add(canvasSizer, 0, wx.CENTER)
        topSizer.Add(buttonsSizer, 0, wx.ALL|wx.EXPAND, 5)

        self.panel.SetSizer(topSizer)
        topSizer.Fit(self)

        # self.Layout()

    def onZoomInBtn(self,event):
        print("ZoomInBtn")

    def onZoomOutBtn(self,event):
        print("ZoomOutBtn")

    def onPrintBtn(self,event):
        print("Print")

########################################################################

class CrocoGui(wx.Frame):

    def __init__(self):

        wx.Frame.__init__(self, None, wx.ID_ANY, title='My Form')

        self.Panel = wx.Panel(self, wx.ID_ANY)

        self.OpenFileBtn = wx.Button(self.Panel, wx.ID_ANY, "Open History File ...")
        self.OpenFileBtn.Bind(wx.EVT_BUTTON, self.onOpenFile)
        self.OpenFileTxt = wx.StaticText(self.Panel, wx.ID_ANY, " ", style=wx.ALIGN_LEFT)

        self.CrocoVariableChoice = wx.Choice(self.Panel, wx.ID_ANY, choices=["Croco Variables ..."])
        self.CrocoVariableChoice.SetSelection(0)
        self.CrocoVariableChoice.Bind(wx.EVT_CHOICE, self.onCrocoVariableChoice)

        self.DerivedVariableChoice = wx.Choice(self.Panel, wx.ID_ANY, choices=["Derived Variables ..."])
        self.DerivedVariableChoice.SetSelection(0)
        self.DerivedVariableChoice.Bind(wx.EVT_CHOICE, self.onDerivedVariableChoice)

        self.ResetColorBtn = wx.Button(self.Panel, wx.ID_ANY, "Reset Color")
        self.ResetColorBtn.Bind(wx.EVT_BUTTON, self.onResetColorBtn)
        self.MinColorTxt = wx.TextCtrl(self.Panel, wx.ID_ANY, "Min Color", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.MinColorTxt.Bind(wx.EVT_TEXT_ENTER, self.onMinColorTxt)
        self.MaxColorTxt = wx.TextCtrl(self.Panel, wx.ID_ANY, "Max Color", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.MaxColorTxt.Bind(wx.EVT_TEXT_ENTER, self.onMaxColorTxt)

        self.LabelTime = wx.StaticText(self.Panel,-1,label="Choose Time",style = wx.ALIGN_CENTER)
        self.TimeMinusBtn = wx.Button(self.Panel, wx.ID_ANY, "<")
        self.TimeMinusBtn.Bind(wx.EVT_BUTTON, self.onTimeMinusBtn)
        self.TimeTxt = wx.TextCtrl(self.Panel, wx.ID_ANY, "Time", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.TimeTxt.Bind(wx.EVT_TEXT_ENTER, self.onTimeTxt)
        self.TimePlusBtn = wx.Button(self.Panel, wx.ID_ANY, ">")
        self.TimePlusBtn.Bind(wx.EVT_BUTTON, self.onTimePlusBtn)

        self.LabelLevel = wx.StaticText(self.Panel,-1,label="Choose level (level>0, depth<0)",style = wx.ALIGN_CENTER)
        self.LevelMinusBtn = wx.Button(self.Panel, wx.ID_ANY, "<")
        self.LevelMinusBtn.Bind(wx.EVT_BUTTON, self.onLevelMinusBtn)
        self.LevelTxt = wx.TextCtrl(self.Panel, wx.ID_ANY, "Level", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.LevelTxt.Bind(wx.EVT_TEXT_ENTER, self.onLevelTxt)
        self.LevelPlusBtn = wx.Button(self.Panel, wx.ID_ANY, ">")
        self.LevelPlusBtn.Bind(wx.EVT_BUTTON, self.onLevelPlusBtn)

        self.LonSectionBtn = wx.Button(self.Panel, wx.ID_ANY, "Longitude Section")
        self.LonSectionBtn.Bind(wx.EVT_BUTTON, self.onLonSectionBtn)
        self.LonSectionTxt = wx.TextCtrl(self.Panel, wx.ID_ANY, "Longitude", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.LonSectionTxt.Bind(wx.EVT_TEXT_ENTER, self.onLonSectionTxt)
        self.LatSectionBtn = wx.Button(self.Panel, wx.ID_ANY, "Latitude Section")
        self.LatSectionBtn.Bind(wx.EVT_BUTTON, self.onLatSectionBtn)
        self.LatSectionTxt = wx.TextCtrl(self.Panel, wx.ID_ANY, "Latitude", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.LatSectionTxt.Bind(wx.EVT_TEXT_ENTER, self.onLatSectionTxt)
        self.HovmullerBtn = wx.Button(self.Panel, wx.ID_ANY, "Hovmuller")
        self.HovmullerBtn.Bind(wx.EVT_BUTTON, self.onHovmullerBtn)
        self.TimeSeriesBtn = wx.Button(self.Panel, wx.ID_ANY, "Time Series")
        self.TimeSeriesBtn.Bind(wx.EVT_BUTTON, self.onTimeSeriesBtn)
        self.VerticalProfileBtn = wx.Button(self.Panel, wx.ID_ANY, "Vertical Profile")
        self.VerticalProfileBtn.Bind(wx.EVT_BUTTON, self.onVerticalProfileBtn)


        # self.PanelCanvas = wx.Panel(self.Panel, wx.ID_ANY)
        self.PanelCanvas = wx.Panel(self.Panel, -1)
        self.figure = Figure(figsize=(figsize[0],figsize[1]))
        # self.figure.canvas.mpl_connect('button_press_event', self.onFigureClick)
        self.canvas = FigureCanvas(self.PanelCanvas, -1, self.figure)
        # self.canvas.mpl_connect('button_press_event', self.onFigureClick)
        # self.canvas.mpl_connect('button_release_event', self.onFigureRelease)
        # self.axes = self.figure.add_axes([0,0,1,1])
        # self.axes = self.figure.add_axes([0.1,0.1,0.9,0.9])

        self.AnimationBtn = wx.Button(self.Panel, wx.ID_ANY, "Animation")
        self.AnimationBtn.Bind(wx.EVT_BUTTON, self.onAnimationBtn)
        self.startTimeTxt = wx.TextCtrl(self.Panel, wx.ID_ANY, "1", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.startTimeTxt.Bind(wx.EVT_TEXT_ENTER, self.onstartTimeTxt)
        self.endTimeTxt = wx.TextCtrl(self.Panel, wx.ID_ANY, "1", style=wx.TE_CENTRE|wx.TE_PROCESS_ENTER)
        self.endTimeTxt.Bind(wx.EVT_TEXT_ENTER, self.onendTimeTxt)
        self.ZoomInBtn = wx.Button(self.Panel, wx.ID_ANY, "Zoom In")
        self.ZoomInBtn.Bind(wx.EVT_BUTTON, self.onZoomInBtn)
        self.ZoomOutBtn = wx.Button(self.Panel, wx.ID_ANY, "Zoom Out")
        self.ZoomOutBtn.Bind(wx.EVT_BUTTON, self.onZoomOutBtn)
        self.PrintBtn = wx.Button(self.Panel, wx.ID_ANY, "Print")
        self.PrintBtn.Bind(wx.EVT_BUTTON, self.onPrintBtn)

        # self.__set_properties()
        self.__do_layout()

        self.sectionFrame = SectionFrame()

        self.currentDirectory = os.getcwd()

    def __do_layout(self):

        topSizer        = wx.BoxSizer(wx.HORIZONTAL)
        leftSizer        = wx.BoxSizer(wx.VERTICAL)
        rightSizer        = wx.BoxSizer(wx.VERTICAL)
        openFileSizer   = wx.BoxSizer(wx.VERTICAL)
        chooseVariablesSizer = wx.BoxSizer(wx.HORIZONTAL)
        colorSizer      = wx.BoxSizer(wx.HORIZONTAL)
        labelTimeSizer  = wx.BoxSizer(wx.HORIZONTAL)
        timeSizer       = wx.BoxSizer(wx.HORIZONTAL)
        labelLevelSizer  = wx.BoxSizer(wx.HORIZONTAL)
        levelSizer       = wx.BoxSizer(wx.HORIZONTAL)
        longitudeSizer  = wx.BoxSizer(wx.HORIZONTAL)
        latitudeSizer   = wx.BoxSizer(wx.HORIZONTAL)
        hovmullerSizer  = wx.BoxSizer(wx.HORIZONTAL)
        timeSeriesSizer = wx.BoxSizer(wx.HORIZONTAL)
        profileSizer   = wx.BoxSizer(wx.HORIZONTAL)
        canvasSizer     = wx.BoxSizer(wx.VERTICAL)
        buttonsSizer    = wx.BoxSizer(wx.HORIZONTAL)

        openFileSizer.Add(self.OpenFileBtn, 0, wx.ALL, 5)
        openFileSizer.Add(self.OpenFileTxt, 1, wx.ALL|wx.EXPAND, 5)
        chooseVariablesSizer.Add(self.CrocoVariableChoice, 0, wx.ALL, 5)
        chooseVariablesSizer.Add(self.DerivedVariableChoice, 0, wx.ALL, 5)

        colorSizer.Add(self.ResetColorBtn, 0, wx.ALL, 5)
        colorSizer.Add(self.MinColorTxt, 0, wx.ALL, 5)
        colorSizer.Add(self.MaxColorTxt, 0, wx.ALL, 5)

        labelTimeSizer.Add(self.LabelTime, 0, wx.ALL|wx.EXPAND, 5)
        timeSizer.Add(self.TimeMinusBtn, 0, wx.ALL, 5)
        timeSizer.Add(self.TimeTxt, 0, wx.ALL, 5)
        timeSizer.Add(self.TimePlusBtn, 0, wx.ALL, 5)

        labelLevelSizer.Add(self.LabelLevel, 0, wx.ALL|wx.EXPAND, 5)
        levelSizer.Add(self.LevelMinusBtn, 0, wx.ALL, 5)
        levelSizer.Add(self.LevelTxt, 0, wx.ALL, 5)
        levelSizer.Add(self.LevelPlusBtn, 0, wx.ALL, 5)

        longitudeSizer.Add(self.LonSectionBtn, 0, wx.ALL, 5)
        longitudeSizer.Add(self.LonSectionTxt, 0, wx.ALL, 5)

        latitudeSizer.Add(self.LatSectionBtn, 0, wx.ALL, 5)
        latitudeSizer.Add(self.LatSectionTxt, 0, wx.ALL, 5)

        hovmullerSizer.Add(self.HovmullerBtn, 0, wx.ALL, 5)

        timeSeriesSizer.Add(self.TimeSeriesBtn, 0, wx.ALL, 5)

        profileSizer.Add(self.VerticalProfileBtn, 0, wx.ALL, 5)

        canvasSizer.Add(self.PanelCanvas, 1, wx.EXPAND , 5)
        buttonsSizer.Add(self.AnimationBtn,0, wx.ALL, 5)
        buttonsSizer.Add(self.startTimeTxt,1, wx.ALL, 5)
        buttonsSizer.Add(self.endTimeTxt,1, wx.ALL, 5)
        buttonsSizer.Add(self.ZoomInBtn,0, wx.ALL, 5)
        buttonsSizer.Add(self.ZoomOutBtn,0, wx.ALL, 5)
        buttonsSizer.Add(self.PrintBtn,0, wx.ALL, 5)

        leftSizer.Add(openFileSizer, 0,wx.ALL|wx.EXPAND, 5 )
        leftSizer.Add(chooseVariablesSizer, 0, wx.ALL|wx.EXPAND, 5)
        leftSizer.Add(labelTimeSizer, 0, wx.ALL|wx.EXPAND, 5)
        leftSizer.Add(timeSizer, 0, wx.ALL|wx.EXPAND, 5)
        leftSizer.Add(labelLevelSizer, 0, wx.ALL|wx.EXPAND, 5)
        leftSizer.Add(levelSizer, 0, wx.ALL|wx.EXPAND, 5)
        leftSizer.Add(longitudeSizer, 0, wx.ALL|wx.EXPAND, 5)
        leftSizer.Add(latitudeSizer, 0, wx.ALL|wx.EXPAND, 5)
        leftSizer.Add(hovmullerSizer, 0, wx.ALL|wx.EXPAND, 5)
        leftSizer.Add(timeSeriesSizer, 0, wx.ALL|wx.EXPAND, 5)
        leftSizer.Add(profileSizer, 0, wx.ALL|wx.EXPAND, 5)
        rightSizer.Add(canvasSizer, 0, wx.EXPAND)
        rightSizer.Add(buttonsSizer, 0, wx.ALL|wx.EXPAND, 5)
        rightSizer.Add(colorSizer, 0, wx.ALL|wx.EXPAND, 5)

        topSizer.Add(leftSizer, 0,wx.ALL|wx.EXPAND, 5 )
        # topSizer.Add(rightSizer, 0,wx.ALL|wx.EXPAND, 5 )
        topSizer.Add(rightSizer, 0,wx.EXPAND, 5 )

        self.Panel.SetSizer(topSizer)
        self.Panel.SetAutoLayout(True)
        topSizer.Fit(self)

        # self.SetAutoLayout(True)
        # self.SetSizer(topSizer)

        self.Layout()


    def onOpenFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
        dlg.Destroy()
        self.croco = Croco(paths[0]) 
        self.OpenFileTxt.SetLabel(paths[0])            
        self.TimeTxt.SetValue(str(self.croco.times[0]))
        self.timeIndex = 0
        self.time = self.croco.times[0]
        self.LevelTxt.SetValue(str(self.croco.crocoGrid.N))
        self.levelIndex=self.croco.crocoGrid.N - 1
        self.startTimeTxt.SetValue(str(self.croco.times[0]))
        self.endTimeTxt.SetValue(str(self.croco.times[-1]))
        self.CrocoVariableChoice.AppendItems(self.croco.ListOfVariables)


    def onFigureClick(self,event):
        self.lonPress, self.latPress = event.xdata, event.ydata
        self.latPressIndex,self.lonPressIndex = self.findLatLonIndex(self.lonPress, self.latPress)
        self.LonSectionTxt.SetValue('%.2F' % self.lonPress)
        self.LatSectionTxt.SetValue('%.2F' % self.latPress)

    def onFigureRelease(self,event):
        self.lonRelease, self.latRelease = event.xdata, event.ydata
        self.lonReleaseIndex,self.latReleaseIndex = self.findLatLonIndex(self.lonRelease, self.latRelease)

    def findLatLonIndex(self, lonValue, latValue):
        ''' Find nearest value is an array '''
        a = abs(self.croco.crocoGrid._lon - lonValue) + \
            abs(self.croco.crocoGrid._lat - latValue)
        return np.unravel_index(a.argmin(),a.shape)
        # idx,idy = np.where(np.abs(array-value)==np.abs(array-value).min())

    def onCrocoVariableChoice(self, event):
        self.variableName = self.CrocoVariableChoice.GetString(self.CrocoVariableChoice.GetSelection())
        # var = self.CrocoVariableChoice.GetCurrentSelection()
        time = str(self.timeIndex)
        level = str(self.levelIndex)
        self.variableXY = self.croco.read_nc(self.variableName, indices= "["+time+","+level+",:,:]")
        self.clim = [np.min(self.variableXY),np.max(self.variableXY)]
        self.mincolor = np.min(self.variableXY)
        self.MinColorTxt.SetValue('%.2E' % self.mincolor)
        self.maxcolor = np.max(self.variableXY)
        self.MaxColorTxt.SetValue('%.2E' % self.maxcolor)
        self.xlim = [np.min(self.croco.crocoGrid._lon),np.max(self.croco.crocoGrid._lon)]
        self.ylim = [np.min(self.croco.crocoGrid._lat),np.max(self.croco.crocoGrid._lat)]
        self.drawxy()

    def onDerivedVariableChoice(self, event):
        self.variableName = self.DerivedVariableChoice.GetString(self.DerivedVariableChoice.GetSelection())
        # time = str(self.timeIndex)
        # level = str(self.levelIndex)
        # self.variableXY = self.croco.read_nc(self.variableName, indices= "["+time+","+level+",:,:]")
        # self.draw()

    def onResetColorBtn(self,event):
        self.clim = [np.min(self.variableXY),np.max(self.variableXY)]
        self.MinColorTxt.SetValue('%.2E' % self.clim[0])
        self.MaxColorTxt.SetValue('%.2E' % self.clim[1])
        self.drawxy()

    def onMinColorTxt(self,event):
        self.clim[0] = float(self.MinColorTxt.GetValue())
        self.drawxy()

    def onMaxColorTxt(self,event):
        self.clim[1] = float(self.MaxColorTxt.GetValue())
        self.drawxy()


    def onTimeMinusBtn(self,event):
        self.timeIndex = max(self.timeIndex - 1,0)
        self.time = self.croco.times[self.timeIndex]
        self.TimeTxt.SetValue(str(self.time))
        self.updateVariableXY()
        self.drawxy()

    def onTimePlusBtn(self,event):
        self.timeIndex = min(self.timeIndex + 1,self.croco.crocoGrid.ntimes - 1)
        self.time = self.croco.times[self.timeIndex]
        self.TimeTxt.SetValue(str(self.time))
        self.updateVariableXY()
        self.drawxy()

    def onTimeTxt(self,event):
        self.time = float(self.TimeTxt.GetValue())
        # find index corresponding to instant time to plot
        self.timeIndex = min( range( len(self.croco.times[:]) ), key=lambda j:abs(time-self.croco.times[j]))
        self.TimeTxt.SetValue(str(self.croco.times[self.timeIndex]))
        self.updateVariableXY()
        self.drawxy()

    def onLevelMinusBtn(self,event):
        self.levelIndex = max(self.levelIndex - 1,0)
        self.LevelTxt.SetValue(str(self.levelIndex + 1))
        self.updateVariableXY()
        self.drawxy()

    def onLevelPlusBtn(self,event):
        self.levelIndex = min(self.levelIndex + 1,self.croco.crocoGrid.N - 1)
        self.LevelTxt.SetValue(str(self.levelIndex + 1))
        self.updateVariableXY()
        self.drawxy()

    def onLevelTxt(self,event):
        time = str(self.timeIndex)
        depth = float(self.LevelTxt.GetValue())
        if depth > 0:
            self.levelIndex = int(self.LevelTxt.GetValue()) - 1
            self.updateVariableXY()
        elif depth < 0:
            zeta = self.croco.read_nc('ssh', indices= "["+time+",:,:]")
            z = self.croco.crocoGrid.scoord2z_r(zeta, alpha=0., beta=0.)
            minlev,maxlev = self.croco.crocoGrid.zslice(None,self.croco.crocoGrid.maskr(),z,depth,findlev=True)
            indices= "["+time+","+str(minlev)+":"+str(maxlev+1)+",:,:]"
            var = self.croco.read_nc(self.variableName, indices=indices)
            dims = self.croco.read_var_dim(self.variableName)
            if "x_u" in dims:
                mask = self.croco.crocoGrid.umask()
                z = self.croco.crocoGrid.rho2u_3d(z)
            elif "y_v" in dims:
                mask = self.croco.crocoGrid.vmask()
                z = self.croco.crocoGrid.rho2v_3d(z)
            else:
                mask = self.croco.crocoGrid.maskr()
            self.variableXY = self.croco.crocoGrid.zslice(var[:,:,:],mask,z[minlev:maxlev+1,:,:],depth)[0]
        else:
            print "baraotrope"
        self.drawxy()

    def onLonSectionBtn(self,event):
        try:
            self.sectionFrame.IsShown()
        except:
            self.sectionFrame = SectionFrame()
        time = str(self.timeIndex)
        lon = str(self.lonPressIndex)
        zeta = self.croco.read_nc('ssh', indices= "["+time+",:,:]")
        self.sectionFrame.y = self.croco.crocoGrid.scoord2z_r(zeta, alpha=0., beta=0)[:,:,self.lonPressIndex]
        self.sectionFrame.variableName = self.variableName
        self.sectionFrame.section = "Longitude"
        self.sectionFrame.latlon = self.lonPress
        self.sectionFrame.time = self.time
        self.sectionFrame.variableZ = self.croco.read_nc(self.variableName, indices= "["+time+",:,:,"+lon+"]")
        self.sectionFrame.x = repmat(self.croco.crocoGrid._lat[:,self.lonPressIndex].squeeze(),self.croco.crocoGrid.N,1)
        self.sectionFrame.drawz()



    def onLonSectionTxt(self,event):
        self.lonPress = float(self.LonSectionTxt.GetValue())
        self.latPressIndex,self.lonPressIndex = self.findLatLonIndex(self.lonPress, self.latPress) 
        try:
            self.sectionFrame.IsShown()
        except:
            self.sectionFrame = SectionFrame()
        time = str(self.timeIndex)
        lon = str(self.lonPressIndex)
        zeta = self.croco.read_nc('ssh', indices= "["+time+",:,:]")
        self.sectionFrame.variableName = self.variableName
        self.sectionFrame.section = "Longitude"
        self.sectionFrame.latlon = self.lonPress
        self.sectionFrame.time = self.time
        self.sectionFrame.y = self.croco.crocoGrid.scoord2z_r(zeta, alpha=0., beta=0)[:,:,self.lonPressIndex]
        self.sectionFrame.variableZ = self.croco.read_nc(self.variableName, indices= "["+time+",:,:,"+lon+"]")
        self.sectionFrame.x = repmat(self.croco.crocoGrid._lat[:,self.lonPressIndex].squeeze(),self.croco.crocoGrid.N,1)
        self.sectionFrame.drawz()


    def onLatSectionBtn(self,event):
        try:
            self.sectionFrame.IsShown()
        except:
            self.sectionFrame = SectionFrame()
        time = str(self.timeIndex)
        lat = str(self.latPressIndex)
        zeta = self.croco.read_nc('ssh', indices= "["+time+",:,:]")
        self.sectionFrame.variableName = self.variableName
        self.sectionFrame.section = "Latitude"
        self.sectionFrame.latlon = self.latPress
        self.sectionFrame.time = self.time
        self.sectionFrame.y = self.croco.crocoGrid.scoord2z_r(zeta, alpha=0., beta=0)[:,self.latPressIndex,:]
        self.sectionFrame.variableZ = self.croco.read_nc(self.variableName, indices= "["+time+",:,"+lat+",:]")
        self.sectionFrame.x = repmat(self.croco.crocoGrid._lon[self.latPressIndex,:].squeeze(),self.croco.crocoGrid.N,1)
        self.sectionFrame.drawz()
        # self.canvas.draw()

    def onLatSectionTxt(self,event):
        self.latPress = float(self.LatSectionTxt.GetValue())
        self.latPressIndex,self.lonPressIndex = self.findLatLonIndex(self.lonPress, self.latPress) 
        print self.lonPressIndex,self.latPressIndex
        try:
            self.sectionFrame.IsShown()
        except:
            self.sectionFrame = SectionFrame()
        time = str(self.timeIndex)
        lat = str(self.latPressIndex)
        zeta = self.croco.read_nc('ssh', indices= "["+time+",:,:]")
        self.sectionFrame.variableName = self.variableName
        self.sectionFrame.section = "Latitude"
        self.sectionFrame.latlon = self.latPress
        self.sectionFrame.time = self.time
        self.sectionFrame.y = self.croco.crocoGrid._scoord2z('r', zeta, alpha=0., beta=0)[0][:,self.latPressIndex,:]
        self.sectionFrame.variableZ = self.croco.read_nc(self.variableName, indices= "["+time+",:,"+lat+",:]")
        self.sectionFrame.x = repmat(self.croco.crocoGrid._lon[self.latPressIndex,:].squeeze(),self.croco.crocoGrid.N,1)
        self.sectionFrame.drawz()


    def onHovmullerBtn(self,event):
        print("Lat Section")

    def onTimeSeriesBtn(self,event):
        lat = str(self.latPressIndex)
        lon = str(self.lonPressIndex)
        level = str(self.levelIndex)
        profile = self.croco.read_nc(self.variableName, indices= "[:,"+level+","+lat+","+lon+"]")
        try:
            self.profileFrame.IsShown()
        except Exception:           
            self.profileFrame = ProfileFrame()
        title="{:s}, Lon={:4.1f}, Lat={:4.1f}, Depth={:4.1f}".\
            format(self.variableName,self.lonPress,self.latPress,\
            self.levelIndex)
        plotCurv(self.profileFrame.figure,profile,title=title)
        self.profileFrame.canvas.draw()
        self.profileFrame.canvas.Refresh()
        self.profileFrame.Show()
        # self.profileFrame.axes = self.figure.add_axes([0.1,0.1,0.9,0.9])
        # self.profileFrame.axes.plot(profile)
        # self.profileFrame.canvas.draw()
        # self.profileFrame.Show()

    def onVerticalProfileBtn(self,event):
        time = str(self.timeIndex)
        lat = str(self.latPressIndex)
        lon = str(self.lonPressIndex)
        title="{:s}, Lon={:4.1f}, Lat={:4.1f}, Time={:4.1f}".\
            format(self.variableName,self.lonPress,self.latPress,\
            self.croco.times[self.timeIndex])
        zeta = self.croco.read_nc('ssh', indices= "["+time+",:,:]")
        z = self.croco.crocoGrid._scoord2z('r', zeta, alpha=0., beta=0)[0][:,self.latPressIndex,self.lonPressIndex]
        profile = self.croco.read_nc(self.variableName, indices= "["+time+",:,"+lat+","+lon+"]")
        try:
            self.profileFrame.IsShown()
        except:
            self.profileFrame = ProfileFrame()
        plotCurv(self.profileFrame.figure,profile,z,title=title,ylabel="depth")
        self.profileFrame.canvas.draw()
        self.profileFrame.canvas.Refresh()
        self.profileFrame.Show()

    def onAnimationBtn(self,event):
        print("Animation")
        try:
            os.makedirs('./Figures')
        except:
            pass

        # for self.timeIndex in range(self.croco.crocoGrid.ntimes):
        for self.timeIndex in range(2,5):
            # time = str(it)
            # level = str(self.levelIndex)
            # self.variableXY = self.croco.read_nc(self.variableName, indices= "["+time+","+level+",:,:]")
            # self.drawxy()
            print str(self.croco.times[self.timeIndex])
            self.TimeTxt.SetValue(str(self.croco.times[self.timeIndex]))
            self.updateVariableXY()
            self.drawxy()
            filename = "./Figures/{:s}.{:04d}.png".format(self.variableName,self.timeIndex)
            self.figure.savefig(filename, dpi=self.figure.dpi)   
        # os.system("ffmpeg -r 5 -i ./Figures/"+self.variableName+".*.png -crf 25 Figures/"+self.variableName+"_xy.mp4")
        # os.system("rm -f ./Figures/"+self.variableName+".*.png")


    def onstartTimeTxt(self,event):
        print("startTimeTxt")

    def onendTimeTxt(self,event):
        print("startTimeTxt")

    def onZoomInBtn(self,event):
        self.xlim = [min(self.lonPress,self.lonRelease),max(self.lonPress,self.lonRelease)]
        self.ylim = [ min(self.latPress,self.latRelease),max(self.latPress,self.latRelease)]
        self.drawxy()

    def onZoomOutBtn(self,event):
        self.xlim = [np.min(self.croco.crocoGrid._lon),np.max(self.croco.crocoGrid._lon)]
        self.ylim = [np.min(self.croco.crocoGrid._lat),np.max(self.croco.crocoGrid._lat)]
        self.drawxy()

    def onPrintBtn(self,event):
        filename = self.variableName + ".png"
        self.figure.savefig(filename, dpi=self.figure.dpi)

    def updateVariableXY(self):
        time = str(self.timeIndex)
        level = str(self.levelIndex)
        self.variableXY = self.croco.read_nc(self.variableName, indices= "["+time+","+level+",:,:]")
        self.drawxy()

    def drawxy(self):
        self.canvas.Destroy()
        self.figure = Figure(figsize=(figsize[0],figsize[1]))
        self.canvas = FigureCanvas(self.PanelCanvas, -1, self.figure)
        self.canvas.mpl_connect('button_press_event', self.onFigureClick)
        self.canvas.mpl_connect('button_release_event', self.onFigureRelease)
        depth = float(self.LevelTxt.GetValue())
        if depth > 0:
            title = "{:s}, Level={:4d}, Time={:4.1f}".format(self.variableName,self.levelIndex,self.croco.times[self.timeIndex])
        else:
            title = "{:s}, Depth={:4.1f}, Time={:4.1f}".format(self.variableName,depth,self.croco.times[self.timeIndex])
        mypcolor(self.figure,self.croco.crocoGrid._lon,self.croco.crocoGrid._lat,self.variableXY,\
                      title=title,\
                      xlabel='Longitude',\
                      ylabel='Latitude',\
                      xlim=self.xlim,\
                      ylim=self.ylim,\
                      clim=self.clim)

        self.canvas.draw()
        self.canvas.Refresh()


# end of class CrocoGui



# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = CrocoGui()
    frame.Show()
    app.MainLoop()