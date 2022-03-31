from pathlib import Path
from gui_helper import generateTSNE, generateUnlabeledImage

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image, ImageTk
import json

with open('config.json') as f:
   config = json.load(f)



OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets")
exitFlag = False

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def exitProgram():
    global exitFlag
    print("Exiting Program")
    exitFlag = True
    exit()
    window.destroy()

global totalCount
totalCount = 0

def countIterator():
    global totalCount
    totalCount = totalCount + 1
    return totalCount

def embedMatplot(fig, col, r):
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=r, column=col, padx=2, pady=2)

def myClick():
    global totalCount
    totalCount = countIterator()

    # clear current matplots and embed new new ones
    plt.clf()
    if (config['images']['enabled'] == True):
        embedMatplot(generateUnlabeledImage(totalCount),0, 0)
    if (config['TSNE']['enabled'] == True):
        embedMatplot(generateTSNE(totalCount),1, 0)

window = Tk()

window.configure(bg = "#FFFFFF")

frame = Frame(window)
frame.grid(row=0,column=0, sticky="n")

myClick()

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

#Radio Button 1
selected_visual = StringVar()
selected_visual.set(' ')
selections = (('Image', 'I'),
         ('TSNE', 'T'),
         ('Histogram', 'H'))

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
scale = (('High Confidence', 'I'),
         ('Moderate Confidence', 'T'),
         ('Low Confidence', 'H'))

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


window.protocol("WM_DELETE_WINDOW", exitProgram)
window.resizable(False, False)
if (exitFlag == False):
    window.mainloop()