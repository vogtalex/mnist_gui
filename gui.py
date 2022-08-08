from pathlib import Path
from csv_gui import initializeCSV, writeToCSV
from visuals_generator import generateUnattackedImage, buildTrajectoryCostReg
from enlarge_visuals_helper import enlargeVisuals, loadFigures
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from functions import generateEpsilonList

imgIdx = 32

with open('config.json') as f:
   config = json.load(f)

outputArray = []
initialLoad = True

ASSETS_PATH = Path(__file__).parent / Path("assets")

maxEpsilon = config["General"]["maxEpsilon"]
epsilonStepSize = config["General"]["epsilonStepSize"]
epsilonList = generateEpsilonList(epsilonStepSize,maxEpsilon)

def embedMatplot(fig, col, r):
    fig.set_size_inches(6, 4)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=r, column=col, padx=2, pady=2)

numRows = 3
numCols = 2
def myClick():
    global imgIdx
    global figureList
    global initialLoad

    if not initialLoad:
        userData = []
        userData.append(entry_1.get())
        userData.append(selected_visual.get())
        userData.append(confidence.get())
        print(userData)
        outputArray.append(userData)
        # clear entry/radio buttons on submission
        entry_1.delete(0,END)
        entry_1.insert(0,"")
        selected_visual.set(None)
        confidence.set(None)
    else:
        initialLoad = False
        if config["TrajectoryRegression"]["enabled"]:
            buildTrajectoryCostReg(imgIdx)

    # clear current matplots and embed new new ones
    plt.clf()
    figureList = loadFigures(epsilonList, imgIdx, maxEpsilon, config)

    # embed all available figures that will fit in specified layout size
    maxFigs = min(numRows*numCols,len(figureList))
    for i in range(numCols):
        if i*numRows>=maxFigs: break
        for j in range(numRows):
            if i*numRows+j>=maxFigs: break
            embedMatplot(figureList[i*numRows+j][0],i,j)

    imgIdx += 1

window = Tk()

window.configure(bg = "#FFFFFF")

frame = Frame(window)
frame.grid(row=0,column=0, sticky="n")

canvas = Canvas(window, bg = "#FFFFFF", height = 800, width = 300, bd = 0, highlightthickness = 0, relief = "ridge")
canvas.grid(row = 0, column = 1)

canvas.create_rectangle(0, 0, 300, 800, fill="#D2D2D2", outline="")
canvas.create_text(12, 85.0, anchor="nw", text="Which visualization\nmost assisted you in\nmaking this decision?", fill="#000000", font=("Roboto", -24))
canvas.create_text(12, 380.0, anchor="nw", text="Prediction Confidence:", fill="#000000", font=("Roboto", -24))
canvas.create_text(12, 24.0, anchor="nw", text="Prediction:", fill="#000000", font=("Roboto", -24))
104,120.5
entry_image_1 = PhotoImage(file = ASSETS_PATH / "entry_1.png")
entry_bg_1 = canvas.create_image(200, 40.5, image=entry_image_1)
entry_1 = Entry(bd=0, bg="#FFFFFF", highlightthickness=0, width=18)
canvas.create_window(200, 40.5, window=entry_1)

button_image_1 = PhotoImage(file = ASSETS_PATH / "button_1.png")
button_1 = Button(image=button_image_1, borderwidth=0, highlightthickness=0, command=(myClick), relief="flat", width=298.19, height=115.15)
canvas.create_window(150, 750, window=button_1)

def enlarge_plots():
    global figureList
    root = Tk()
    p1 = enlargeVisuals(root, figureList)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

button_2 = Button(command=(enlarge_plots), width= 40, height = 3, text= "Enlarge Visualizations")
canvas.create_window(150, 660, window=button_2)

if config["General"]["showOriginal"]:
    def orig_image():
        fig = generateUnattackedImage(imgIdx-1)
        fig.show()

    button_3 = Button(command=(orig_image), width= 40, height = 3, text= "Original Image")
    canvas.create_window(150, 600, window=button_3)

#Radio Button 1
selected_visual = StringVar()
selected_visual.set(' ')

selections = []
if config['Images']['enabled']:
    selections.append(('Image', 'Image'))
if config["BoxPlot"]["enabled"]:
    selections.append(('Box Plot', 'Box Plot'))
if config["TSNE"]["enabled"]:
    selections.append(('TSNE', 'TSNE'))
if config["Histogram"]["enabled"]:
    selections.append(('Histogram', 'Histogram'))
if config["TrajectoryRegression"]["enabled"]:
    selections.append(('Attack Reconstruction', 'Attack Reconstruction'))

height = 200
for visual in selections:
    r = Radiobutton(
        window,
        text=visual[0],
        value=visual[1],
        variable=selected_visual,
        anchor=W,
        justify = LEFT,
        bg="#D2D2D2",
        font=("Roboto", -18)
    )
    canvas.create_window(20, height, anchor=W, window=r)
    height += 30

#Radio Button 2
confidence = StringVar()
confidence.set(' ')
scale = (('High Confidence', 'High Confidence'),
         ('Moderate Confidence', 'Moderate Confidence'),
         ('Low Confidence', 'Low Confidence'))

height = 440
for x in scale:
    r2 = Radiobutton(
        window,
        text=x[0],
        value=x[1],
        variable=confidence,
        anchor=W,
        justify = LEFT,
        bg="#D2D2D2",
        font=("Roboto", -18)
    )
    canvas.create_window(20, height, anchor=W, window=r2)
    height += 30

def exitProgram():
    print("Exiting Program")
    initializeCSV()
    writeToCSV(outputArray)
    exit()
    window.destroy()

myClick()
window.protocol("WM_DELETE_WINDOW", exitProgram)
window.resizable(False, False)
window.mainloop()
