#Import the required libraries
from tkinter import *
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from updated_tsne import generateUnlabeledImage, generateTSNEPlots, generateHistograms, generateBoxPlot, generateUnattackedImage

def embedMatplot(fig, col, r):
    fig.set_size_inches(15, 7)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=r, column=col, padx=2, pady=2, columnspan=2)

def nextPlot():
    global currPlot
    currPlot += 1
    findPlot()

def lastPlot():
    global currPlot
    currPlot -= 1
    findPlot()

def findPlot():
    global currPlot
    global totalCount
    global figureList

    if currPlot >= 7:
        currPlot = 0
        findPlot()
    if currPlot < 0:
        currPlot = 6
        findPlot()
    
    embedMatplot(figureList[currPlot],0,0)


def loadFigures(epsilonList):
    figureList = []
    for eps in epsilonList:
        temp, _ = generateHistograms(totalCount, eps)
        figureList.append(temp)
    figureList.append(generateBoxPlot(totalCount))
    figureList.append(generateHistograms(totalCount, 10))
    return figureList
    

def exitProgram():
    global exitFlag
    print("Exiting Program")
    exitFlag = True
    exit()

root = Tk()   

totalCount = 0
exitFlag = False
currPlot = 0
epsilonList = [0,2,4,6,8]
figureList = loadFigures(epsilonList)

temp, _ = generateHistograms(0, 0)
embedMatplot(temp,0, 0)

button_1 = Button(
    command=(lastPlot),
    width= 40,
    height = 3,
    text= "Back"
)

button_1.grid(row=1,column=0)

button_2 = Button(
    command=(nextPlot),
    width= 40,
    height = 3,
    text= "Next"
)

button_2.grid(row=1,column=1)

root.protocol("WM_DELETE_WINDOW", exitProgram)
if (exitFlag == False):
    root.mainloop()   

