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
    figureList.append(generateHistograms(imgIdx, max(epsilonList)+1)[0]) # all epsilons histogram
    figureList.append(generateBoxPlot(imgIdx))
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
  def __init__(self, master, figureList):
    self.root = master
    self.currPlot = 0
    self.figureList = figureList
    self.currentEmbed = None
    self.maxPlots = 6 +1

    # generate initial plot
    self.embedPlot(self.figureList[self.currPlot])

    # back button
    button_1 = Button(master = self.root, command=lambda:self.nextPlot(-1), width= 40, height = 3, text= "Back")
    button_1.grid(row=1,column=0)
    # next button
    button_2 = Button(master = self.root, command=lambda:self.nextPlot(1), width= 40, height = 3, text= "Next")
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
    self.embedPlot(self.figureList[self.currPlot])

  def nextPlot(self, dir):
    # go to next plot based on which button was pressed, wrapping around if it goes below 0 or above max
    self.currPlot = (self.currPlot + dir) % self.maxPlots
    self.embedPlot(self.figureList[self.currPlot])

  def embedPlot(self, fig):
    # delete canvas each time, as creating new FigureCanvasTkAgg's can cause big slowdown when closing window
    temp = None
    if self.currentEmbed:
        temp = self.currentEmbed

    # set max width/height based on screensize and dpi
    fig.set_size_inches(self.root.winfo_screenwidth()/self.root.winfo_fpixels('1i'), self.root.winfo_screenheight()/self.root.winfo_fpixels('1i')-1)
    self.currentEmbed = Frame(self.root)
    canvas = FigureCanvasTkAgg(fig, master = self.currentEmbed)
    canvas.draw()
    canvas.get_tk_widget().pack()
    self.currentEmbed.grid(row=0, column=0, padx=2, pady=2, columnspan=4)

    if temp:
        temp.destroy()
