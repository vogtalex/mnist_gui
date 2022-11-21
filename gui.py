from pathlib import Path
from csv_gui import initializeCSV, writeToCSV
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import time

from functions import generateEpsilonList, AutoScrollbar
from visuals_generator import buildTrajectoryCostReg, getTrueLabel, getAttackStrength
from enlarge_visuals_helper import enlargeVisuals, loadFigures

# define how many examples of each visualization type are wanted
# (all visualizations w/ feedback, all visualizations no feedback, no visualizations no feedback)
visualization_split = [2,10,5,10]

# rows/cols of canvases to show on main page
numRows = 2
numCols = 2

with open('config.json') as f:
   config = json.load(f)

transitions = [sum(visualization_split[:i+1]) for i in range(len(visualization_split))]
current_mode = 0

outputArray = []
initialLoad = True

ASSETS_PATH = Path(__file__).parent / Path("assets")

imgIdx = config["General"]["startIdx"]
maxEpsilon = config["General"]["maxEpsilon"]
epsilonStepSize = config["General"]["epsilonStepSize"]
epsilonList = generateEpsilonList(epsilonStepSize,maxEpsilon)

def popup(msg,parent):
    popup = Toplevel()
    popup.wm_title("!")
    label = Label(popup, text=msg, font="Roboto")
    label.pack(side="top", fill="x", pady=10)
    B1 = Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.protocol("WM_DELETE_WINDOW", lambda: popup.destroy())

    parent.wait_window(popup)

def myClick():
    global imgIdx,figureList,initialLoad,startTime,current_mode

    if not initialLoad:
        endTime = time.time()
        # verify the user has entered all currently visible fields
        if entry_1.get()=="" or current_mode<2 and (confidence.get()=="None" or any([item[1].get()==-1 for item in selections])):
            return

        userData = []
        # get all data values and append them to the output data array
        userData.append(getTrueLabel(imgIdx))
        userData.append(entry_1.get())
        userData.append(getAttackStrength(imgIdx))
        userData.append(config["General"]["displaySubset"])
        userData.append(imgIdx)
        userData.append(endTime-startTime)
        userData.append(confidence.get())
        for item in selections:
            userData.append(item[1].get())
        print(userData)
        outputArray.append(userData)

        # clear entry/radio buttons
        entry_1.delete(0,END)
        entry_1.insert(0,"")
        for item in selections:
            item[1].set(-1)
        confidence.set(None)

        if current_mode==0:
            popup(f"The previous example was a: {int(getTrueLabel(imgIdx))}",window)

        imgIdx += 1

        # if image index passes subset gracefully exit program
        if imgIdx > config["Model"]["subsetSize"]:
            popup("End of experiment run",window)
            exitProgram()

        # if user has hit threshold for current mode, transition and update what's active on the screen.
        if current_mode < len(transitions) and imgIdx > transitions[current_mode]-1:
            current_mode+=1
            if current_mode==1:
                print("Exiting training mode")
            elif current_mode==2:
                print("switching to no feedback")
                # remove elements from canvas which are no longer necessary
                for element in canvasDelete:
                    canvas.delete(element)
            elif current_mode==3:
                print("switching to no visualizations")
                # disable visualizations in config and remove matplotlib canvases
                config["BoxPlot"]["enabled"]=False
                config["TSNE"]["enabled"]=False
                config["TrajectoryRegression"]["enabled"]=False
                config["Histogram"]["enabled"]=False
                while len(frame.winfo_children())>1:
                    frame.winfo_children()[1].destroy()
            elif current_mode==4:
                popup("End of experiment run",window)
                exitProgram()

        figureList = loadFigures(epsilonList, imgIdx, maxEpsilon, config)
        startTime = time.time()

    # on initial load
    else:
        # create title row for data
        userData = []
        userData.append("True label")
        userData.append("User prediction")
        userData.append("Attack strength")
        userData.append("Subset")
        userData.append("Index")
        userData.append("Time taken (in s)")
        userData.append("User confidence")
        for item in selections:
            userData.append((item[0]).replace('\n',' '))
        print(userData)
        outputArray.append(userData)

        # build the trajectoryCostReg function with the starting index
        if config["TrajectoryRegression"]["enabled"]:
            buildTrajectoryCostReg(imgIdx)

        figureList = loadFigures(epsilonList, imgIdx, maxEpsilon, config)

        # embed all available figures that will fit in specified layout size
        maxFigs = min(numRows*numCols,len(figureList))
        for i in range(numCols):
            if i*numRows>=maxFigs: break
            for j in range(numRows):
                if i*numRows+j>=maxFigs: break
                # embed matplotlib figure
                figureCanvas = FigureCanvasTkAgg(figureList[i*numRows+j][0], master=frame)
                figureCanvas.draw()
                figureCanvas.get_tk_widget().grid(row=j, column=i, padx=2, pady=2)

        startTime = time.time()
        initialLoad = False

# create tkinter window
window = Tk()

# create selections array based on which are active in config
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
    item[1].set(-1)

window.configure(bg = "#FFFFFF")

# create scrollbar & canvas for scrolling
horzScrollBar = AutoScrollbar(window, orient=HORIZONTAL)
horzScrollBar.grid(row = 1, column = 0, stick='sew')

scrollCanvas = Canvas(window, xscrollcommand = horzScrollBar.set, width = 604*min(2,numCols,1+len(selections)), height=800)
scrollCanvas.grid(row=0, column=0, sticky='nsew')

horzScrollBar.config(command = scrollCanvas.xview)

global frame
frame = Frame(scrollCanvas)
frame.grid(row=0,column=0, sticky="n")

# populate matplotlib canvases
myClick()

# add frame w/ visualizations to canvas
scrollCanvas.create_window(0, 0, anchor=NW, window=frame)
frame.update_idletasks()

# Configuring scrollable area of canvas
scrollCanvas.config(scrollregion=scrollCanvas.bbox("all"))

# if scrollbar is shown, bind vertical scroll to scroll bar
scrollBarShown = min(numRows*numCols,len(figureList))>4
if scrollBarShown:
    scrollCanvas.bind_all('<MouseWheel>', lambda event: scrollCanvas.xview_scroll(int(-1*(event.delta/120)), "units"))

canvas = Canvas(window, bg = "#FFFFFF", height = 800 + 24*scrollBarShown, width = 300, bd = 0, highlightthickness = 0, relief = "ridge")
canvas.grid(row = 0, column = 1, stick='ns', rowspan = 1 + scrollBarShown)

# create user prediction entry
canvas.create_rectangle(0, 0, 300, 800 + 24*scrollBarShown, fill="#D2D2D2", outline="")
canvas.create_text(12, 24.0, anchor="nw", text="Prediction:", fill="#000000", font=("Roboto", -24))
104,120.5
entry_image_1 = PhotoImage(file = ASSETS_PATH / "entry_1.png")
entry_bg_1 = canvas.create_image(200, 40.5, image=entry_image_1)
entry_1 = Entry(canvas, bd=0, bg="#FFFFFF", highlightthickness=0, width=8,font=("segoe-ui 18"))
canvas.create_window(200, 38.5, window=entry_1)

# create submit button
button_image_1 = PhotoImage(file = ASSETS_PATH / "button_1.png")
button_1 = Button(canvas, image=button_image_1, borderwidth=0, highlightthickness=0, command=(myClick), relief="flat", width=298.19, height=115.15)
canvas.create_window(150, 750 + 24*scrollBarShown, window=button_1)

# create enlarged view popup & button to open it
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

# store canvas elements being created for removal later
canvasDelete = []

# if visualizations are enabled, create feedback questions
if len(selections) > 0:
    canvasDelete.append(canvas.create_text(12, 80.0, anchor="nw", text="How significantly did \nthe visualizations impact \nyour decision:", fill="#000000", font=("Roboto", -24), justify=LEFT))

    # helpfulness labels
    width = 105
    height = 180
    for radioLabel in ['Not very\nimpactful','Somewhat\nimpactful','Very\nimpactful']:
        canvasDelete.append(canvas.create_text(width, height, anchor="n", text=radioLabel, fill="#000000", font=("Roboto", -18), justify = CENTER))
        width+=80

    # create all radiobuttons & labels for each visualization
    height += 70
    for visual in selections:
        l = Label(window,text=visual[0],anchor=W,justify = LEFT,bg="#D2D2D2",font=("Roboto", -18))
        canvasDelete.append(canvas.create_window(1, height, anchor=W, window=l))

        for i in range(3):
            r = Radiobutton(window, value = i+1, variable=visual[1], bg="#D2D2D2")
            idx = canvas.create_window(108 + i*80, height + 2, window=r)
            canvasDelete.append(idx)
            canvas.tag_lower(idx)

        height += 35

# create prediction confidence radio buttons & labels
height = 480
canvasDelete.append(canvas.create_text(12, height, anchor="nw", text="Prediction Confidence:", fill="#000000", font=("Roboto", -24)))
confidence = StringVar()
confidence.set(None)
scale = (('High Confidence', 'High Confidence'),
         ('Moderate Confidence', 'Moderate Confidence'),
         ('Low Confidence', 'Low Confidence'))

height+=50
for x in scale:
    r2 = Radiobutton(window, text=x[0], value=x[1], variable=confidence, anchor=W, justify = LEFT, bg="#D2D2D2", font=("Roboto", -18))
    canvasDelete.append(canvas.create_window(20, height, anchor=W, window=r2))
    height += 30

def exitProgram():
    print("Exiting Program")
    # on exit, intitialize a csv file and write the stored data to it
    initializeCSV()
    writeToCSV(outputArray)
    exit()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", exitProgram)
window.bind('<Control-w>', lambda e:exitProgram())
window.resizable(False, False)
window.mainloop()
