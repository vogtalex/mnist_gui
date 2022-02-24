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


def loadImg():
    global img
    Tk().withdraw()
    imgName = askopenfilenames(title="Select file", filetypes=[(
        "Picture files", ("*.jpg", "*.png", "*.gif", "*.ppm", "*.ico"))])
    images = []
    for currName in imgName:
        if str(currName) is not ".":
            try:
                img = Image.open(currName)
                img = img.convert("RGB")
            except:
                print("Error", "The image you tried to open was corrupted.")
                return
            images.append(np.array(img))
    return images


images = loadImg()
save('data.npy', images)
