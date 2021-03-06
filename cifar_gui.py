from pathlib import Path
from winreg import HKEY_LOCAL_MACHINE

from torch import histogram
from csv_gui import initializeCSV, writeToCSV
from visuals_generator_cifar import generateUnlabeledImage, generateHistograms, generateBoxPlot, generateUnattackedImage
from enlarge_visuals_helper import enlargeVisuals, loadFigures, loadFiguresCifar

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image, ImageTk
import json

global totalCount
totalCount = 400

histogramEpsilon = 4


with open('config.json') as f:
   config = json.load(f)

eps = config['Histogram']['weightDir']
histogramEpsilon = int(eps[1])
histogramEpsilon *= 2
outputArray = []

epsilonList = [0,2,4,6,8]

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets")
exitFlag = False
ChangeFigsFlag = False

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def exitProgram():
    global exitFlag
    print("Exiting Program")
    exitFlag = True
    initializeCSV()
    writeToCSV(outputArray)
    exit()
    window.destroy()

def countIterator():
    global totalCount
    totalCount = totalCount + 1
    return totalCount

def getFigureList():
    global figureList
    return figureList

def getChangeFigsFlag():
    global ChangeFigsFlag
    return ChangeFigsFlag

def setChangeFigsFlag():
    global ChangeFigsFlag
    ChangeFigsFlag = False

def embedMatplot(fig, col, r):
    fig.set_size_inches(6, 4)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=r, column=col, padx=2, pady=2)

def myClick():
    global totalCount
    global figureList
    global p1
    global ChangeFigsFlag
    totalCount = countIterator()
    if totalCount != 1:
        userData = []
        userData.append(entry_1.get())
        userData.append(selected_visual.get())
        userData.append(confidence.get())
        print(userData)
        outputArray.append(userData)

    # clear current matplots and embed new new ones
    plt.clf()
    figureList = loadFiguresCifar(epsilonList, totalCount)
    if (config['Images']['enabled'] == True):
        embedMatplot(generateUnlabeledImage(totalCount),0, 0)
    if (config['Cifar']['enabled'] == True):
        newEps = int(histogramEpsilon / 2)
        temp = figureList[newEps]
        embedMatplot(temp,1, 0)
    if (config['Cifar']['enabled'] == True):
        embedMatplot(figureList[6],0, 1)
    if (config['Cifar']['enabled'] == True):
        embedMatplot(figureList[5],1, 1)
    ChangeFigsFlag = True

def enlarge_plots():
    global figureList
    # global ChangeFigsFlag

    root = Tk()
    p1 = enlargeVisuals(0, root, figureList)
    root.protocol("WM_DELETE_WINDOW", p1.exitProgram)
    # if (p1.ChangeFigsFlag == True):
    #     currFigs = getFigureList()
    #     p1.updateFigs(currFigs)
    #     setChangeFigsFlag()
    #     print("Updating the Figs")
    if (p1.exit_flag == False):
        root.mainloop()  

    print("Enlarged plot")

def orig_image():
    fig = generateUnattackedImage(totalCount)
    fig.show()


window = Tk()

window.configure(bg = "#FFFFFF")

frame = Frame(window)
frame.grid(row=0,column=0, sticky="n")

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 800,
    width = 300,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.grid(row = 0, column = 1)

canvas.create_rectangle(
    0,
    0,
    300,
    800,
    fill="#D2D2D2",
    outline="")

canvas.create_text(
    12,
    185.0,
    anchor="nw",
    text="Which visualization\nassisted you in making \nthis decision?",
    fill="#000000",
    font=("Roboto", 24 * -1)
)

canvas.create_text(
    12,
    419.0,
    anchor="nw",
    text="Prediction Confidence:",
    fill="#000000",
    font=("Roboto", 24 * -1)
)

canvas.create_text(
    12,
    104.0,
    anchor="nw",
    text="Prediction:",
    fill="#000000",
    font=("Roboto", 24 * -1)
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    200,
    120.5,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0
)

canvas.create_window(200, 120.5, window=entry_1)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=(myClick),
    relief="flat",
    width=298.1923828125,
    height=115.14483642578125
)


canvas.create_window(150, 750, window=button_1)

button_2 = Button(
    command=(enlarge_plots),
    width= 40,
    height = 3,
    text= "Enlarge Visualizations"
)

canvas.create_window(150, 660, window=button_2)

button_3 = Button(
    command=(orig_image),
    width= 40,
    height = 3,
    text= "Original Image"
)

canvas.create_window(150, 600, window=button_3)

#Radio Button 1
selected_visual = StringVar()
selected_visual.set(' ')
selections = (('Image', 'Image'),
         ('TSNE', 'TSNE'),
         ('Histogram', 'Histogram'))

height = 300
for visual in selections:
    r = Radiobutton(
        window,
        text=visual[0],
        value=visual[1],
        variable=selected_visual,
        anchor=W,
        justify = LEFT,
        bg="#D2D2D2",
        font=("Roboto", 18 * -1)
    )
    canvas.create_window(20, height, anchor=W, window=r)
    height += 30


#Radio Button 2
confidence = StringVar()
confidence.set(' ')
scale = (('High Confidence', 'High Confidence'),
         ('Moderate Confidence', 'Moderate Confidence'),
         ('Low Confidence', 'Low Confidence'))

height = 480
for x in scale:
    r2 = Radiobutton(
        window,
        text=x[0],
        value=x[1],
        variable=confidence,
        anchor=W,
        justify = LEFT,
        bg="#D2D2D2",
        font=("Roboto", 18 * -1)
    )
    canvas.create_window(20, height, anchor=W, window=r2)
    height += 30


myClick()

window.protocol("WM_DELETE_WINDOW", exitProgram)
window.resizable(False, False)
if (exitFlag == False):
    window.mainloop()