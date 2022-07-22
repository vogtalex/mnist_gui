from pathlib import Path
from csv_gui import initializeCSV, writeToCSV
from visuals_generator import generateUnlabeledImage, generateTSNEPlots, generateHistograms, generateBoxPlot, generateUnattackedImage
from enlarge_visuals_helper import enlargeVisuals, loadFigures
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

imgIdx = 552
initialLoad = True

histogramEpsilon = 2

with open('config.json') as f:
   config = json.load(f)

eps = config['Histogram']['weightDir']
histogramEpsilon = 2 * int(eps[1])
outputArray = []

ASSETS_PATH = Path(__file__).parent / Path("assets")

#epsilonList = [0,2,4,6,8]
epsilonList = list(range(5))

def embedMatplot(fig, col, r):
    fig.set_size_inches(6, 4)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=r, column=col, padx=2, pady=2)

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

    imgIdx += 1

    # clear current matplots and embed new new ones
    plt.clf()
    figureList = loadFigures(epsilonList, imgIdx)
    # image
    if (config['Images']['enabled'] == True):
        embedMatplot(generateUnlabeledImage(imgIdx),0, 0)
    # specific epsilon histogram
    if (config['MNIST']['enabled'] == True):
        newEps = int(histogramEpsilon / 2)
        embedMatplot(figureList[newEps],1, 0)
    # all epsilon histogram
    if (config['MNIST']['enabled'] == True):
        embedMatplot(figureList[6],0, 1)
    # boxplot
    if (config['MNIST']['enabled'] == True):
        embedMatplot(figureList[5],1, 1)

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

entry_image_1 = PhotoImage(file = ASSETS_PATH / "entry_1.png")
entry_bg_1 = canvas.create_image(
    200,
    120.5,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    width=18
)

canvas.create_window(200, 120.5, window=entry_1)

button_image_1 = PhotoImage(file = ASSETS_PATH / "button_1.png")
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

def enlarge_plots():
    global figureList
    root = Tk()
    p1 = enlargeVisuals(0, root, figureList)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

button_2 = Button(
    command=(enlarge_plots),
    width= 40,
    height = 3,
    text= "Enlarge Visualizations"
)

canvas.create_window(150, 660, window=button_2)

def orig_image():
    fig = generateUnattackedImage(imgIdx)
    fig.show()

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

def exitProgram():
    print("Exiting Program")
    initializeCSV()
    writeToCSV(outputArray)
    exit()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", exitProgram)
window.resizable(False, False)
window.mainloop()
