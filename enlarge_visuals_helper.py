#Import the required libraries
from tkinter import *
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from updated_tsne import generateUnlabeledImage, generateTSNEPlots, generateHistograms, generateBoxPlot, generateUnattackedImage


def loadFigures(epsilonList, totalCount):
    figureList = []
    for eps in epsilonList:
        temp, _ = generateHistograms(totalCount, eps)
        figureList.append(temp)
    figureList.append(generateBoxPlot(totalCount))
    figureList.append(generateHistograms(totalCount, 10))
    return figureList



class enlargeVisuals():
  def __init__(self, idx, master, figureList):
    self.root = master
    self.index = idx
    self.exit_flag = False
    self.currPlot = 0
    self.epsilonList = [0,2,4,6,8]
    self.figureList = figureList

    temp, _ = generateHistograms(self.index, 0)
    self.embedPlot(temp,0, 0)

    button_1 = Button(
        master = self.root,
        command=(self.lastPlot),
        width= 40,
        height = 3,
        text= "Back"
    )

    button_1.grid(row=1,column=0)

    button_2 = Button(
        master = self.root,
        command=(self.nextPlot),
        width= 40,
        height = 3,
        text= "Next"
    )

    button_2.grid(row=1,column=3)

    self.period_slider = Scale(  master = self.root,
                            from_=0, to_=16, 
                            resolution=0.10,
                            orient='horizontal', 
                            length= 500, 
                            width = 30, 
                            label = "X-Axis Lower Bound",
                            command=self.updateXAxis)
    self.period_slider.set(0)
    self.period_slider.grid(row=1, column = 1)

    self.period_slider2 = Scale(  master = self.root,
                            from_=0, to_=16, 
                            resolution=0.10,
                            orient='horizontal', 
                            length= 500, 
                            width = 30, 
                            label = "X-Axis Upper Bound",
                            command=self.updateXAxis)
    self.period_slider2.set(16)
    self.period_slider2.grid(row=1, column = 2)

  def exitProgram(self):
    print("Exiting Program")
    self.exit_flag = True
    self.root.destroy()
    # exit()

  def updateXAxis(self, temp):
    x = self.period_slider.get()
    y = self.period_slider2.get()
    
    for fig in self.figureList:
        for ax in fig.axes:
            ax.set_xlim(x, y)
    self.genPlots()

  def genPlots(self):
    if self.currPlot >= 7:
        self.currPlot = 0
        self.genPlots()
    if self.currPlot < 0:
        self.currPlot = 6
        self.genPlots()
    self.embedPlot(self.figureList[self.currPlot],0,0)

  def nextPlot(self):
    self.currPlot += 1
    self.genPlots()

  def lastPlot(self): 
    self.currPlot -= 1
    self.genPlots()

  def embedPlot(self, fig, col, r):
    fig.set_size_inches(15, 7)
    canvas = FigureCanvasTkAgg(fig, master = self.root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=r, column=col, padx=2, pady=2, columnspan=4)


# root = Tk()
# p1 = enlargeVisuals(0, root)
# root.protocol("WM_DELETE_WINDOW", p1.exitProgram)
# if (p1.exit_flag == False):
#     root.mainloop()  
