from models.net import *
from csv_gui import *
import torch
import numpy as np
import matplotlib.pyplot as plt

import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
import torch.onnx as onnx
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision

import tkinter as tk
from tkinter import *

from PIL import Image, ImageTk

from functools import partial

from numpy import load


images = load('data.npy', allow_pickle=True)


# Generates an image of epsilon 0.15
def generateNewImage(count):
    label, new_label, image, = images[count]
    plt.title("What is this number?")
    plt.imshow(image, cmap="gray")
    plt.savefig('saved_figure2.png')
    return label

# Initializes the height and width of image for GUI
HEIGHT = 200
WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200

# Variables initialized for my user path. Can be changed for different user
path1 = "saved_figure.png"
path2 = "saved_figure2.png"
path3 = "saved_figure3.png"


# Count of image displayed currently and count guessed correctly in the GUI
correctCount = 0
global totalCount 
totalCount = 0

# Iterates the total count to iterate through images


def countIterator():
    global totalCount
    totalCount = totalCount + 1
    return totalCount


def quitFunction():
    root.destroy()
    quit()

# Generates new model prediction
def modelPrediction(totalCount):
    stringModel = "Model Prediction: "
    ga, answer, gar, = images[totalCount]
    convAnswer = str(answer)
    stringModel = stringModel + convAnswer
    def_label = tk.Label(visual_aid_frame, text=stringModel)
    def_label.pack(padx=10, pady=5, fill=tk.BOTH)

#Opens image and places new image on gui
def openImage():
    image1 = Image.open(path2)
    test = ImageTk.PhotoImage(image1, master=root)
    label1 = tk.Label(image_frame, image=test)
    label1.image = test
    label1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)


# Function for button for user guess
def myClick():
    # Create new image
    global totalCount
    global label

    if totalCount == 0 :
        label = generateNewImage(totalCount)
        modelPrediction(totalCount)
    
    currCount = correctCount
    newLabel = label
    currNum = e.get()

    #Add guess to CSV
    writeDataCSV(int(currNum), newLabel)

    if (int(currNum) == newLabel):
        currCount = currCount + 1
        myLabel = Label(output_frame, text="Correct!")
        myLabel.pack(padx=10, pady=5, fill=tk.BOTH)
    else:
        myLabel2 = Label(output_frame, text="Incorrect")
        myLabel2.pack(padx=10, pady=5, fill=tk.BOTH)
    
    totalCount = countIterator()
    label = generateNewImage(totalCount)
    openImage()





#Initialize CSV by deleting prior csv "response.csv"
initializeCSV()



# GUI
root = Tk()
root.title("Human Testing of Adversarial Training")

# Setup frames
# global image_frame
image_frame = tk.Frame(root, background="#FFF0C1", bd=1, relief="sunken")
input_frame = tk.Frame(root, background="#D2E2FB", bd=1, relief="sunken")
visual_aid_frame = tk.Frame(root, background="#CCE4CA", bd=1, relief="sunken")
output_frame = tk.Frame(root, background="#F5C2C1", bd=1, relief="sunken")
number_frame = tk.Frame(root, background="#0000FF", bd=1, relief="sunken")
image_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
input_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
visual_aid_frame.grid(row=0, column=1, rowspan=2,
                      sticky="nsew", padx=2, pady=2)
output_frame.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=2, pady=2)


# Configure frames
root.grid_rowconfigure(0, weight=3)
root.grid_rowconfigure(1, weight=2)
root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=2)
root.grid_columnconfigure(2, weight=2)


# Create a photoimage object of the image in the path
generateNewImage(0)
image1 = Image.open(path2)
test = ImageTk.PhotoImage(image1, master=root)
Label(image_frame, image=test).grid(
    row=0, column=0, sticky="nsew", padx=2, pady=2)

# Creates entry box for user guess
e = Entry(input_frame, width=50)
e.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)


# Adds a Button
myButton = Button(input_frame, text="Click Me!",
                  pady=50, command=partial(myClick))
myButton.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

# Visual Aid
stringModel = "Model Prediction: "
ga, answer, gar, = images[totalCount]
convAnswer = str(answer)
stringModel = stringModel + convAnswer
def_label = tk.Label(visual_aid_frame, text=stringModel)
def_label.pack(padx=10, pady=5, fill=tk.BOTH)

exit_button = Button(root, text="Exit", command=root.quit)
exit_button.grid(row=3, column=0, pady=20)

# Loop
root.mainloop()

#Output CSV
formatCSV()
outputCSV()