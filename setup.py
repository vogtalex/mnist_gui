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
import os
from numpy import save

GUI_HEIGHT = 300
GUI_WIDTH = 500
BACKGROUND_COLOR = None

def loadImg(config):
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

    save("data.npy", images)
    config["outputDir"] = os.path.join(os.getcwd(), "data.npy")

# options setup
options = {"images":{"function":loadImg, "buttonText":"Upload images"},
    "TSNE":{"function":None, "buttonText":"Upload something"},
    "Generic":{"function":None, "buttonText":"Generic select"}}

# from https://stackoverflow.com/a/44687752/17215948
def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")

def on_quit():
    global exitFlag
    exitFlag = True
    root.destroy()

# create output keys for dictionary
for option, config in options.items():
    config["enabled"] = False
    config["outputDir"] = None

class setup(tk.Tk):
    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)
        self.configure(bg=BACKGROUND_COLOR)
        self.title("Visualization setup")

        # set window size based
        self.geometry(str(GUI_WIDTH)+"x"+str(GUI_HEIGHT))

        label = Label(self, text="Select which visualizations you want to enable:")
        label.grid(row=0,column=0)

        options_frame = Frame(self)
        options_frame.grid()

        # used to track which options the user selected to enable
        self.enabled = {}

        # create checkboxes for each option in options dictionary
        for option, config in options.items():
            is_selected = BooleanVar()
            cb = Checkbutton(options_frame, text=option, variable=is_selected, onvalue=1, offvalue=0)
            self.enabled[option] = is_selected
            cb.grid()

        submit = Button(self, text="submit", command = lambda: self.set_enabled())
        submit.grid()

        # handles user Xing out of window
        self.protocol("WM_DELETE_WINDOW", on_quit)

    def set_enabled(self):
        for option, config in options.items():
            config["enabled"] = self.enabled[option].get()
        self.destroy()

class setupOptions(tk.Tk):
    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)

        # Additional setup for enabled visualizationsself
        self.configure(bg=BACKGROUND_COLOR)
        self.title("Additional setup")

        # used to track paths of options for each visualization
        self.paths = {}

        curr_row=0
        for option, config in options.items():
            if(config["enabled"]):
                label = Label(self, text = config["buttonText"])
                label.grid(row = curr_row, column=0)

                # call function in dict
                button = Button(self, text="upload", command = lambda config=config: config["function"](config))

                button.grid(row = curr_row, column=1)

                curr_row+=1

        # (paths[option] =
        finish = Button(self, text="finish", command = lambda: self.set_paths())
        finish.grid()

        # handles user Xing out of window
        self.protocol("WM_DELETE_WINDOW", on_quit)

    def set_paths(self):
        for visualization, path in self.paths.items():
            options[visualization][outputDir] = path

        for option, config in options.items():
            config.pop("function")
            config.pop("buttonText")

        self.destroy()

exitFlag = False
root = setup()
root.mainloop()

print(options)

if not exitFlag:
    root = setupOptions()
    root.mainloop()
    print(options)
