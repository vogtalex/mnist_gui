#Import the required libraries
from tkinter import *
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from visuals_generator import generateUnlabeledImage, generateTSNEPlots, generateHistograms, generateBoxPlot, generateUnattackedImage
from visuals_generator_cifar import generateHistograms as cifar_hist
from visuals_generator_cifar import generateBoxPlot as cifar_box

def loadFigures(epsilonList, imgIdx):
    figureList = []
    for eps in epsilonList:
        figureList.append(generateHistograms(imgIdx, eps)[0])
    figureList.append(generateBoxPlot(imgIdx))
    figureList.append(generateHistograms(imgIdx, max(epsilonList)+1)[0])
    return figureList

def loadFiguresCifar(epsilonList, imgIdx):
    figureList = []
    for eps in epsilonList:
        temp, _ = cifar_hist(imgIdx, eps)
        figureList.append(temp)
    figureList.append(cifar_box(imgIdx))
    figureList.append(cifar_hist(imgIdx, 10))
    return figureList

class enlargeVisuals():
  def __init__(self, idx, master, figureList):
    self.root = master
    self.currPlot = 0
    self.epsilonList = [0,2,4,6,8]
    self.figureList = figureList
    self.currentEmbed = None

    # generate initial plot
    self.genPlots()

    # create back/next buttons
    button_1 = Button(master = self.root, command=self.lastPlot, width= 40, height = 3, text= "Back")
    button_1.grid(row=1,column=0)

    button_2 = Button(master = self.root, command=self.nextPlot, width= 40, height = 3, text= "Next")
    button_2.grid(row=1,column=3)

    # create x limit sliders
    self.period_slider = Scale(master = self.root, from_=0, to_=16, resolution=0.50, orient='horizontal', length= 500, width = 30, label = "X-Axis Lower Bound", command=self.updateXAxis)
    self.period_slider.set(0)
    self.period_slider.grid(row=1, column = 1)

    self.period_slider2 = Scale(master = self.root, from_=0, to_=16, resolution=0.50, orient='horizontal', length= 500, width = 30, label = "X-Axis Upper Bound", command=self.updateXAxis)
    self.period_slider2.set(16)
    self.period_slider2.grid(row=1, column = 2)

  def updateXAxis(self, temp):
    for fig in self.figureList:
        for ax in fig.axes:
            # get x limits from sliders and update limits for all subplots
            ax.set_xlim(self.period_slider.get(), self.period_slider2.get())
    self.genPlots()

  def genPlots(self):
    # loop around plot if above max # or below 0
    if self.currPlot > 6:
        self.currPlot = 0
    elif self.currPlot < 0:
        self.currPlot = 6
    self.embedPlot(self.figureList[self.currPlot])

  def nextPlot(self):
    self.currPlot += 1
    self.genPlots()

  def lastPlot(self):
    self.currPlot -= 1
    self.genPlots()

  def embedPlot(self, fig):
    fig.set_size_inches(15, 7)
    # clear current canvas so window doesn't take a long time to close (may not be working)
    if self.currentEmbed:
        self.currentEmbed.get_tk_widget().delete("all")
    self.currentEmbed = FigureCanvasTkAgg(fig, master = self.root)
    self.currentEmbed.draw()
    self.currentEmbed.get_tk_widget().grid(row=0, column=0, padx=2, pady=2, columnspan=4)
