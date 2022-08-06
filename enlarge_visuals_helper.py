from tkinter import Button, Frame, Scale
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from visuals_generator import generateUnlabeledImage, generateTSNEPlots, generateHistograms, generateBoxPlot, trajectoryCostReg

def loadFigures(epsilonList, imgIdx, maxEpsilon, config):
    figureList = []
    # second value in tuple is whether x limits can be modified
    if config['Images']['enabled']:
        figureList.append((generateUnlabeledImage(imgIdx),False))
    if config["BoxPlot"]["enabled"]:
        figureList.append((generateBoxPlot(imgIdx),False))
    if config["TSNE"]["enabled"]:
        figureList.append((generateTSNEPlots(imgIdx),False))
    if config["Histogram"]["enabled"]:
        # all epsilons histogram, generates if epsilon val is greater than max
        figureList.append((generateHistograms(imgIdx, maxEpsilon+1)[0],True))
        for eps in epsilonList:
            figureList.append((generateHistograms(imgIdx, eps)[0],True))
    if config["TrajectoryRegression"]["enabled"]:
        figureList.append((trajectoryCostReg(imgIdx),False))
    return figureList

class enlargeVisuals():
  def __init__(self, master, figureList):
    self.root = master
    self.currPlot = 0
    self.figureList = figureList
    self.currentEmbed = None
    self.maxPlots = len(figureList)

    # generate initial plot
    self.embedPlot(self.figureList[self.currPlot][0])

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

  def updateXAxis(self,_):
    # only update x limits if enabled for visualization
    if self.figureList[self.currPlot][1]:
        # get x limits from sliders and update limits for all subplots
        xMin = self.period_slider.get()
        xMax = self.period_slider2.get()
        for ax in self.figureList[self.currPlot][0].axes:
            ax.set_xlim(xMin, xMax)
        self.embedPlot(self.figureList[self.currPlot][0])

  def nextPlot(self, dir):
    # go to next plot based on which button was pressed, wrapping around if it goes below 0 or above max
    self.currPlot = (self.currPlot + dir) % self.maxPlots
    if self.figureList[self.currPlot][1]:
        self.updateXAxis(0)
    else:
        self.embedPlot(self.figureList[self.currPlot][0])

  def embedPlot(self, fig):
    temp = self.currentEmbed if self.currentEmbed else None

    # set max width/height based on screensize and dpi
    fig.set_size_inches(self.root.winfo_screenwidth()/self.root.winfo_fpixels('1i'), self.root.winfo_screenheight()/self.root.winfo_fpixels('1i')-1)
    self.currentEmbed = Frame(self.root)
    canvas = FigureCanvasTkAgg(fig, master = self.currentEmbed)
    canvas.draw()
    canvas.get_tk_widget().pack()
    self.currentEmbed.grid(row=0, column=0, padx=2, pady=2, columnspan=4)

    # delete old canvas each time, as creating new FigureCanvasTkAgg's can cause big slowdown when closing window
    if temp:
        temp.destroy()
