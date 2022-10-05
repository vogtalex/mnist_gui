from pathlib import Path
from csv_gui import initializeCSV, writeToCSV
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

from functions import generateEpsilonList, AutoScrollbar
from visuals_generator import buildTrajectoryCostReg, getTrueLabel, getAttackStrength
from enlarge_visuals_helper import enlargeVisuals, loadFigures

with open('config.json') as f:
   config = json.load(f)

outputArray = []
initialLoad = True

ASSETS_PATH = Path(__file__).parent / Path("assets")

imgIdx = config["General"]["startIdx"]
maxEpsilon = config["General"]["maxEpsilon"]
epsilonStepSize = config["General"]["epsilonStepSize"]
epsilonList = generateEpsilonList(epsilonStepSize,maxEpsilon)

def embedMatplot(fig, col, r):
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=r, column=col, padx=2, pady=2)

numRows = 2
numCols = 2
def myClick():
    global imgIdx
    global figureList
    global initialLoad

    if not initialLoad:
        userData = []
        userData.append(getTrueLabel(imgIdx))
        userData.append(entry_1.get())
        for item in selections:
            userData.append(item[1].get())
        userData.append(confidence.get())
        userData.append(getAttackStrength(imgIdx))
        userData.append(config["General"]["displaySubset"])
        userData.append(imgIdx)
        print(userData)
        outputArray.append(userData)

        # clear entry/radio buttons on submission
        entry_1.delete(0,END)
        entry_1.insert(0,"")
        for item in selections:
            item[1].set(None)
        confidence.set(None)

        imgIdx += 1

        figureList = loadFigures(epsilonList, imgIdx, maxEpsilon, config)
    else:
        userData = []
        userData.append("True label")
        userData.append("User prediction")
        for item in selections:
            userData.append((item[0]).replace('\n',' '))
        userData.append("User confidence")
        userData.append("Attack strength")
        userData.append("Subset")
        userData.append("Index")
        print(userData)
        outputArray.append(userData)


        initialLoad = False
        if config["TrajectoryRegression"]["enabled"]:
            buildTrajectoryCostReg(imgIdx)

        figureList = loadFigures(epsilonList, imgIdx, maxEpsilon, config)

        # embed all available figures that will fit in specified layout size
        maxFigs = min(numRows*numCols,len(figureList))
        for i in range(numCols):
            if i*numRows>=maxFigs: break
            for j in range(numRows):
                if i*numRows+j>=maxFigs: break
                embedMatplot(figureList[i*numRows+j][0],i,j)

window = Tk()

selections = []
if config["BoxPlot"]["enabled"]:
    selections.append(('Box Plot', IntVar()))
if config["TSNE"]["enabled"]:
    selections.append(('TSNE', IntVar()))
if config["Histogram"]["enabled"]:
    selections.append(('Histogram', IntVar()))
if config["TrajectoryRegression"]["enabled"]:
    selections.append(('Attack\nRegression', IntVar()))

for item in selections:
    item[1].set(None)

window.configure(bg = "#FFFFFF")

horzScrollBar = AutoScrollbar(window, orient=HORIZONTAL)
horzScrollBar.grid(row = 1, column = 0, stick='sew')

scrollCanvas = Canvas(window, xscrollcommand = horzScrollBar.set, width = 604*min(2,numCols,1+len(selections)), height=800)
scrollCanvas.grid(row=0, column=0, sticky='nsew')

horzScrollBar.config(command = scrollCanvas.xview)

global frame
frame = Frame(scrollCanvas)
frame.grid(row=0,column=0, sticky="n")

myClick()

# add frame w/ visualizations to canvas
scrollCanvas.create_window(0, 0, anchor=NW, window=frame)

# Calling update_idletasks method
frame.update_idletasks()

# Configuring canvas
scrollCanvas.config(scrollregion=scrollCanvas.bbox("all"))

scrollBarShown = min(numRows*numCols,len(figureList))>4
if scrollBarShown:
    # bind vertical scroll to horizontal scroll bar
    scrollCanvas.bind_all('<MouseWheel>', lambda event: scrollCanvas.xview_scroll(int(-1*(event.delta/120)), "units"))

canvas = Canvas(window, bg = "#FFFFFF", height = 800 + 24*scrollBarShown, width = 300, bd = 0, highlightthickness = 0, relief = "ridge")
canvas.grid(row = 0, column = 1, stick='ns', rowspan = 1 + scrollBarShown)

canvas.create_rectangle(0, 0, 300, 800 + 24*scrollBarShown, fill="#D2D2D2", outline="")
canvas.create_text(12, 24.0, anchor="nw", text="Prediction:", fill="#000000", font=("Roboto", -24))
104,120.5
entry_image_1 = PhotoImage(file = ASSETS_PATH / "entry_1.png")
entry_bg_1 = canvas.create_image(200, 40.5, image=entry_image_1)
entry_1 = Entry(canvas, bd=0, bg="#FFFFFF", highlightthickness=0, width=8,font=("segoe-ui 18"))
canvas.create_window(200, 38.5, window=entry_1)

button_image_1 = PhotoImage(file = ASSETS_PATH / "button_1.png")
button_1 = Button(canvas, image=button_image_1, borderwidth=0, highlightthickness=0, command=(myClick), relief="flat", width=298.19, height=115.15)
canvas.create_window(150, 750 + 24*scrollBarShown, window=button_1)

def enlarge_plots():
    global figureList
    root = Tk()
    enlargeVisuals(root, figureList)
    root.protocol("WM_DELETE_WINDOW", lambda: root.destroy())
    root.bind('<Control-w>', lambda e: root.destroy())
    root.resizable(height=True, width=False)
    root.mainloop()

button_2 = Button(canvas, command=(enlarge_plots), width= 40, height = 3, text= "Enlarge Visualizations")
canvas.create_window(150, 660 + 24*scrollBarShown, window=button_2)

if len(selections) > 0:
    canvas.create_text(12*6, 80.0, anchor="nw", text="Helpfulness of\nvisualizations:", fill="#000000", font=("Roboto", -24), justify=CENTER)

    width = 105
    for radioLabel in ['Not very\nhelpful','Somewhat\nhelpful','Very\nhelpful']:
        canvas.create_text(width, 140, anchor="n", text=radioLabel, fill="#000000", font=("Roboto", -18), justify = CENTER)
        width+=80

    height = 210
    for visual in selections:
        l = Label(window,text=visual[0],anchor=W,justify = LEFT,bg="#D2D2D2",font=("Roboto", -18))
        canvas.create_window(1, height, anchor=W, window=l)

        for i in range(3):
            r = Radiobutton(window, value = i+1, variable=visual[1], bg="#D2D2D2")
            idx = canvas.create_window(108 + i*80, height + 2, window=r)
            canvas.tag_lower(idx)

        height += 35

#Radio Button 2
height = 480
canvas.create_text(12, height, anchor="nw", text="Prediction Confidence:", fill="#000000", font=("Roboto", -24))
confidence = StringVar()
confidence.set(None)
scale = (('High Confidence', 'High Confidence'),
         ('Moderate Confidence', 'Moderate Confidence'),
         ('Low Confidence', 'Low Confidence'))

height+=50
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

window.protocol("WM_DELETE_WINDOW", exitProgram)
window.bind('<Control-w>', lambda e:exitProgram())
window.resizable(False, False)
window.mainloop()
