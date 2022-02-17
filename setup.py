import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.filedialog import askopenfilenames
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np
import json
import webbrowser
from numpy import save

GUI_HEIGHT = 300
GUI_WIDTH = 500
BACKGROUND_COLOR = None

def loadImg():
    global img
    Tk().withdraw()
    imgName = askopenfilenames(title = "Select file",filetypes = [("Picture files",("*.jpg","*.png","*.gif","*.ppm","*.ico"))])
    images = []
    for currName in imgName:
        if str(currName) != ".":
            try:
                img = Image.open(currName)
                img = img.convert("RGB")
            except:
                print("Error","The image you tried to open was corrupted.")
                return
            images.append(np.array(img))
    return images

# images = loadImg()
# save('data.npy', images)

options = {"images":{"enabled":False, "function":loadImg},
    "TSNE":{"enabled":False, "function":None},
    "Generic":{"enabled":False, "function":None}}

# used to track which options the user selected to enable
enabled = {}

# from https://stackoverflow.com/a/44687752/17215948
def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")

def set_enabled():
    for option, config in options.items():
        config["enabled"] = enabled[option].get()

root = Tk()
root.configure(bg=BACKGROUND_COLOR)
root.title("Visualization setup")

# set window size based
root.geometry(str(GUI_WIDTH)+"x"+str(GUI_HEIGHT))

label = Label(root, text="Select which visualizations you want to enable:")
label.grid(row=0,column=0)

options_frame = Frame(root)
options_frame.grid()

# create checkboxes for each option in options dictionary
for option, config in options.items():
    print(config)
    is_selected = BooleanVar()
    cb = Checkbutton(options_frame, text=option, variable=is_selected, onvalue=1, offvalue=0)
    enabled[option] = is_selected
    cb.grid()

submit = Button(root, text="submit", command = set_enabled)
submit.grid()

# handles user Xing out of window
root.protocol("WM_DELETE_WINDOW",root.quit)

# start options window
root.mainloop()

print(options)
