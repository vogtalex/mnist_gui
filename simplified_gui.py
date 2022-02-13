from ast import Global
from pyexpat import model
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

# Variables initialized for my dataset. Can be changed for different user
path1 = "saved_figure.png"
imageTitle = "What is this number?"

#This variable changes if user has labeled data or not. Change it to false if you don't have labeled data
labeledData = False

#This variable changes if user has model predictions. Change it to false if you don't have model predictions
modelData = False

# Initializes the height and width of image for GUI
HEIGHT = 200
WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200

# Count of image displayed currently and count guessed correctly in the GUI
global totalCount 
totalCount = 0

# Generates an unlabeled image
def generateUnlabeledImage(count):
    image = images[count]
    plt.title(imageTitle)
    plt.imshow(image, cmap="gray")
    plt.savefig(path1)


# Iterates the total count to iterate through images
def countIterator():
    global totalCount
    totalCount = totalCount + 1
    return totalCount

#Opens image and places new image on gui
def openImage():
    image1 = Image.open(path1)
    test = ImageTk.PhotoImage(image1, master=root)
    label1 = tk.Label(image_frame, image=test)
    label1.image = test
    label1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

def myUnlabeledClick():
    # Create new image
    global totalCount

    if totalCount == 0 :
        generateUnlabeledImage(totalCount)
    
    currNum = e.get()

    #Add guess to CSV
    writeDataCSV(currNum, currNum)

    totalCount = countIterator()
    generateUnlabeledImage(totalCount)
    openImage()


#Initialize CSV by deleting prior csv "response.csv"
initializeCSV()

# GUI
root = Tk()
root.title("Human Testing of Adversarial Training")

# Setup frames
# global image_frame
image_frame = tk.Frame(root, background="#FFFFFF", bd=1, relief="sunken")
input_frame = tk.Frame(root, background="#FFFFFF", bd=1, relief="sunken")
QA_frame = tk.Frame(root, background="#FFFFFF", bd=1, relief="sunken")
image_frame.grid(row=0, column=0, padx=2, pady=2)
input_frame.grid(row=1, column=0, padx=2, pady=2)
QA_frame.grid(row=1, column=1, padx=2, pady=2)


# Configure frames
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=0)

# Create a photoimage object of the image in the path
generateUnlabeledImage(0)
image1 = Image.open(path1)
test = ImageTk.PhotoImage(image1, master=root)
Label(image_frame, image=test).grid(
    row=0, column=0, sticky="nsew", padx=2, pady=2)

# Creates entry box for user guess
e = Entry(input_frame, width=50, justify=CENTER, font = 20)
e.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
e.insert(0, "Enter your guess here")


# Adds a Button
myButton = Button(input_frame,
                  text="Submit Prediction",
                  pady=50,
                  height= 3,
                  width = 50,
                  font=20,
                  command=partial(myUnlabeledClick))
myButton.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)



exit_button = Button(root, text="Exit",
                    command=root.quit, 
                    height = 3, 
                    width = 50, 
                    background= '#D11A2A', 
                    fg= 'white',
                    font=50)
exit_button.grid(row=2, column=0, pady=20)


# Loop
root.mainloop()

#Format CSV from user input
formatCSV()

#QA
root2 = Tk()
root2.title("Survey:")

questions = [
    "On a scale from 1 - 5, how certain did you feel about your answers?",
    "Do the visualizations influence your decision in determining the image?",
    "Would you recommend the utilization of this tool?"
]

mult_choice = {
        "yes": 1,
        "no": 0
    }

scale = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5
    }

v = [tk.IntVar(root2, idx) for idx in range(3)]
QA_Array = []

for idx, text in enumerate(questions):

        tk.Label(root2, text=text, wraplength=270, justify='left').pack(padx=20)

        if idx == 0:
            for choice, value in scale.items():
                button = tk.Radiobutton(root2, text=choice, variable=v[idx], value=value, justify='left')
                button.pack(padx=(20,0))
        else:
            for choice, value in mult_choice.items():
                tk.Radiobutton(root2, text=choice, variable=v[idx], value=value).pack(padx=(30,0))

#Currently broken. Doesn't produce corret values from radio buttons
for x in range(3):
    QA_Array.append(v[x].get())

exit_button = Button(root2, text="Exit",
                    command=root.quit, 
                    height = 3, 
                    width = 50, 
                    background= '#D11A2A', 
                    fg= 'white',
                    font=50)
exit_button.pack(pady=2)

root2.mainloop()

writeToCSV_QA(QA_Array)