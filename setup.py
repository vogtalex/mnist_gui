import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.filedialog import askopenfilenames, askdirectory
from tkinter import simpledialog
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np
import json
import webbrowser
import os
from functools import partial
from numpy import save

GUI_HEIGHT = 300
GUI_WIDTH = 500
BACKGROUND_COLOR = None
OUTPUT_FILE_NAME = "config.json"

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

def loadModel(config):
    config.pop("outputDir",None)
    Tk().withdraw()
    model = askopenfilenames(title="Select model",filetypes = [("Python file","*.py")])
    weights = askopenfilenames(title="Select model weights",filetypes = [("Pytorch model weight file","*.pth")])
    # paths are saved as tuples. Converts first element of tuple into the needed path and then converts to string for saving to json
    config["modelDir"]=str(Path(model[0]))
    config["weightDir"]=str(Path(weights[0]))

def loadTSNE(config):
    config.pop("outputDir",None)
    Tk().withdraw()
    weights = simpledialog.askstring(title = "Select Epsilon", prompt = "Enter epsilon (Example: e0)")
    # paths are saved as tuples. Converts first element of tuple into the needed path and then converts to string for saving to json
    config["weightDir"]=str(Path(weights))

def loadTSNEIMG(config):
    config.pop("outputDir",None)
    Tk().withdraw()
    weights = askdirectory(title="Select image file directory")
    # paths are saved as tuples. Converts first element of tuple into the needed path and then converts to string for saving to json
    config["weightDir"]=str(Path(weights))



# options. Functions to be added have to be defined above
options = {"Images":{"function":loadImg, "buttonText":"Upload images manually"},
    "TSNE":{"function":loadTSNEIMG, "buttonText":"Upload image file directory"},
    "Histogram":{"function":loadTSNE, "buttonText":"Upload test image file directory"},
    "BoxPlot":{"function":None, "buttonText":"Generate boxplot data"},
    "Generic":{"function":None, "buttonText":"Generic select"}}

# function to run when Xing out of either setup window
def on_quit():
    global exitFlag
    exitFlag = True
    root.destroy()

# create keys used for output in dictionary
for option, config in options.items():
    config["enabled"] = False
    config["outputDir"] = None

# define exitFlag
exitFlag = False

class setup(tk.Tk):
    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)
        self.configure(bg=BACKGROUND_COLOR)
        self.title("Visualization setup")

        # set window size based on constants
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

    # set enable status of options based on enabled dictionary
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
            # only create button/label if that option is enabled
            if(config["enabled"]):
                label = Label(self, text = option+": "+config["buttonText"])
                label.grid(row = curr_row, column=0, sticky=W)

                # call function from options dict
                button = Button(self, text="upload", command = partial(self.runFunction, config))

                button.grid(row = curr_row, column=1)

                curr_row+=1

        finish = Button(self, text="finish", command = lambda: self.set_paths())
        finish.grid()

        # handles user Xing out of window
        self.protocol("WM_DELETE_WINDOW", on_quit)

    def runFunction(self, config):
        if config["function"] != None:
            config["function"](config)

    def set_paths(self):
        # set paths of enabled visualizations
        for visualization, path in self.paths.items():
            options[visualization][outputDir] = path

        # remove dictionary entries only used for setup
        for option, config in options.items():
            config.pop("function")
            config.pop("buttonText")

        self.quit()

# selecting which visualizations to use
root = setup()
root.mainloop()

# if user didn't X out of window, run next setup window
if not exitFlag:
    # add
    root = setupOptions()
    root.mainloop()

# if user didn't X out of either window, save to file.
if not exitFlag:
    with open(OUTPUT_FILE_NAME, 'w') as fp:
        json.dump(options, fp, indent=4)

# example of opening the json file
f=open(OUTPUT_FILE_NAME)
test=json.load(f)
print(test)
