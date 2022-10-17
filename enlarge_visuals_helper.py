from tkinter import Button, Frame, Scale
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from visuals_generator import generateUnlabeledImage, generateTSNEPlots, generateHistograms, generateBoxPlot, trajectoryCostReg, generateUnattackedImage
import time
import pickle

# load enabled figures
def loadFigures(epsilonList, imgIdx, maxEpsilon, config):
    figureList = []
    # second value in tuple is whether x limits can be modified
    if config['Images']['enabled']:
        figureList.append((generateUnlabeledImage(imgIdx),False))
    if config["BoxPlot"]["enabled"]:
        figureList.append((generateBoxPlot(imgIdx),False))
    if config["TSNE"]["enabled"]:
        figureList.append((generateTSNEPlots(imgIdx),False))
    if config["TrajectoryRegression"]["enabled"]:
        figureList.append((trajectoryCostReg(imgIdx),False))
    if config["Histogram"]["enabled"]:
        # all epsilons histogram, generates if epsilon val is greater than max
        allEpsFig, maxHeight = generateHistograms(imgIdx, maxEpsilon+1)
        figureList.append((allEpsFig,True))
        for eps in epsilonList:
            figureList.append((generateHistograms(imgIdx, eps, maxHeight),True))
    if config["General"]["showOriginal"]:
        figureList.append((generateUnattackedImage(imgIdx),False))
    return figureList

class enlargeVisuals():
  def __init__(self, master, figureList):
    # variable initialization
    self.root = master
    self.currPlot = 0
    self.figureList = figureList
    self.currentEmbed = None
    self.maxPlots = len(figureList)

    # generate initial plot
    self.embedPlot(self.figureList[self.currPlot][0])

    # create back/next buttons
    button_1 = Button(master = self.root, command=lambda:self.nextPlot(-1), width= 40, height = 3, text= "Back")
    button_1.grid(row=1,column=0)
    self.root.bind("<Left>",lambda e:self.nextPlot(-1))

    button_2 = Button(master = self.root, command=lambda:self.nextPlot(1), width= 40, height = 3, text= "Next")
    button_2.grid(row=1,column=3)
    self.root.bind("<Right>",lambda e:self.nextPlot(1))

    # create x limit sliders. bind updateXAxis to mouse release so it's only called when slider is released
    self.period_slider = Scale(master = self.root, from_=5, to_=16, resolution=0.50, orient='horizontal', length= 500, width = 30, label = "X-Axis Lower Bound", takefocus=False)
    self.period_slider.bind("<ButtonRelease-1>", self.updateXAxis)
    self.period_slider.set(0)
    self.period_slider.grid(row=1, column = 1)

    self.period_slider2 = Scale(master = self.root, from_=5, to_=16, resolution=0.50, orient='horizontal', length= 500, width = 30, label = "X-Axis Upper Bound", takefocus=False)
    self.period_slider2.bind("<ButtonRelease-1>", self.updateXAxis)
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
    # update plots x axis if allowed in visualization creation, else just embed it
    if self.figureList[self.currPlot][1]:
        self.updateXAxis(0)
    else:
        self.embedPlot(self.figureList[self.currPlot][0])

  def embedPlot(self, fig):
    # pickle/unpickle to make the figure a new figure to avoid this bug: https://github.com/matplotlib/matplotlib/issues/23809
    fig = pickle.loads(pickle.dumps(fig))
    temp = self.currentEmbed if self.currentEmbed else None

    # if figure is TSNE increase the size of points in the figure
    if fig.get_label() == 'TSNE':
        for scatter in fig.get_axes()[0].collections:
            scatter.set_sizes([10])
        fig.get_axes()[1].collections[0].set_sizes([5])
        fig.get_axes()[1].collections[1].set_sizes([25])

    scaler = 1.1
    # set max width/height based on screensize and dpi
    fig.set_size_inches((self.root.winfo_screenwidth()/self.root.winfo_fpixels('1i')-1)/scaler, (self.root.winfo_screenheight()/self.root.winfo_fpixels('1i')-1)/scaler)
    self.currentEmbed = Frame(self.root)
    canvas = FigureCanvasTkAgg(fig, master = self.currentEmbed)
    canvas.draw()
    canvas.get_tk_widget().pack()
    self.currentEmbed.grid(row=0, column=0, padx=2, pady=2, columnspan=4)

    # delete old canvas each time, as creating new FigureCanvasTkAgg's can cause big slowdown when closing window
    if temp:
        temp.destroy()
