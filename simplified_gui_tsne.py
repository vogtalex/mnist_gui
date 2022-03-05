from ast import Global
from pyexpat import model
from models.net import *
from csv_gui import *
from tsne_run import *
import torch
import numpy as np
import matplotlib.pyplot as plt

import time
import tkinter as tk
from tkinter import messagebox as mb
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
import json

from numpy import load

npys = './npys'
eps = 'e3'
examples = 'examples'
limit = 10000

images_orig = np.load(os.path.join(npys, examples, 'advdata.npy')
                      ).astype(np.float64)[:limit]

# images_orig = np.load('./npys/advdata.npy').astype(np.float64)
images = []
for i in range(len(images_orig)):
    images.append(images_orig[i].reshape(28, 28))

# Variables initialized for my dataset. Can be changed for different user
path1 = "saved_image.png"
path2 = "tsne_output.png"
imageTitle = "What is this number?"

# This variable changes if user has labeled data or not. Change it to false if you don't have labeled data
labeledData = False

# This variable changes if user has model predictions. Change it to false if you don't have model predictions
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
    return plt.gcf()

# Iterates the total count to iterate through images
def countIterator():
    global totalCount
    totalCount = totalCount + 1
    return totalCount

# embeds the visualization in the appropriate column
def embedMatplot(fig, col):
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=col, padx=2, pady=2)

def myUnlabeledClick():
    # Create new image
    global totalCount

    currNum = e.get()

    # Add guess to CSV
    writeToCSV(currNum)

    totalCount = countIterator()

    # clear current matplots and embed new new ones
    plt.clf()
    embedMatplot(generateUnlabeledImage(totalCount),0)
    embedMatplot(generateTSNE(totalCount),1)

# Initialize CSV by deleting prior csv "response.csv"
initializeCSV()

# GUI
root = Tk()
root.title("Human Testing of Adversarial Training")

# Setup frames
# global image_frame
image_frame = tk.Frame(root, background="#FFFFFF", bd=1, relief="sunken")
input_frame = tk.Frame(root, bd=1, relief="sunken")
image_frame.grid(row=0, column=0, padx=2, pady=2)
input_frame.grid(row=1, column=0, padx=2, pady=2, columnspan=2)

# Configure frames
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=0)

# Create a photoimage object of the image in the path
embedMatplot(generateUnlabeledImage(0),0)
embedMatplot(generateTSNE(0),1)

# Creates entry box for user guess
lbl = Label(input_frame, text="What does this image depict?", font=20)
lbl.grid(row=0, column=0, sticky="nsew", padx=5, pady=20)

e = Entry(input_frame, width=30, justify=CENTER, font=20)
e.grid(row=0, column=1, sticky="nsew", padx=5, pady=20)
e.insert(0, "Enter your guess here")

lbl = Label(input_frame, text="Which visualizations led you to this answer?", justify=LEFT, font=20)
lbl.grid(row=1, column=0, sticky="nsew", padx=5, pady=20)

a = Entry(input_frame, width=30, justify=CENTER, font=20)
a.grid(row=1, column=1, sticky="nsew", padx=5, pady=20)

# Adds a Button
myButton = Button(input_frame,
                  text="Submit",
                  height=3,
                  width=30,
                  font=20,
                  background='#343a40',
                  fg='white',
                  command=partial(myUnlabeledClick))
myButton.grid(row=2, column=1, pady=20)

exit_button = Button(root, text="Exit",
                     command=root.quit,
                     height=3,
                     width=50,
                     background='#D11A2A',
                     fg='white',
                     font=50)
exit_button.grid(row=3, column=0, pady=20, columnspan=2)

root.configure(background="white")

# Loop
root.protocol("WM_DELETE_WINDOW", root.destroy)
root.mainloop()

# Format CSV from user input
# formatCSV()

# # QA
# root2 = Tk()
# root2.title("Survey:")

# questions = [
#     "On a scale from 1 - 5, how certain did you feel about your answers?",
#     "Do the visualizations influence your decision in determining the image?",
#     "Would you recommend the utilization of this tool?"
# ]

# mult_choice = {
#     "yes": 1,
#     "no": 0
# }

# scale = {
#     "1": 1,
#     "2": 2,
#     "3": 3,
#     "4": 4,
#     "5": 5
# }

# # v = [tk.IntVar(root2, idx) for idx in range(3)]
# var = IntVar()
# var1 = IntVar()
# var2 = IntVar()

# for idx, text in enumerate(questions):

#     tk.Label(root2, text=text, wraplength=270, justify='left').pack(padx=20)

#     if idx == 0:
#         for choice, value in scale.items():
#             Radiobutton(root2, text=choice, variable=var,
#                         value=value, justify='left').pack(padx=(30, 0))

#     if idx == 1:
#         for choice, value in mult_choice.items():
#             Radiobutton(root2, text=choice,
#                         variable=var1, value=value).pack(padx=(30, 0))
#     if idx == 2:
#         for choice, value in mult_choice.items():
#             Radiobutton(root2, text=choice,
#                         variable=var2, value=value).pack(padx=(30, 0))

# # Currently broken. Doesn't produce corret values from radio buttons
# # for x in range(3):
# #     QA_Array.append(v[x].get())

# print(var)

# exit_button = Button(root2, text="Exit",
#                      command=root.quit,
#                      height=3,
#                      width=50,
#                      background='#D11A2A',
#                      fg='white',
#                      font=50)
# exit_button.pack(pady=2)

# x = var.get()
# y = var1.get()
# z = var2.get()

# root2.mainloop()

# writeToCSV_QA(str(x))
# writeToCSV_QA(str(y))
# writeToCSV_QA(str(z))

# Uses code from https://www.geeksforgeeks.org/python-mcq-quiz-game-using-tkinter/

class Quiz:
    def __init__(self):
        self.qno = 0
        self.disp_title()
        self.disp_ques()
        self.opt_sel = IntVar()
        self.opts = self.radio_buttons()
        self.disp_opt()
        self.buttons()
        self.total_size = len(question)
        self.correct = 0

    def disp_res(self):
        wrong_count = self.total_size - self.correct
        correct = f"Correct: {self.correct}"
        wrong = f"Wrong: {wrong_count}"

        score = int(self.correct / self.total_size * 100)
        result = f"Score: {score}%"

        mb.showinfo("Result", f"{result}\n{correct}\n{wrong}")

    def check_ans(self, qno):
        writeToCSV_QA(str(self.opt_sel.get()))
        if self.opt_sel.get() == answer[qno]:
            return True

    def next_btn(self):
        if self.check_ans(self.qno):
            self.correct += 1

        self.qno += 1

        if self.qno == self.total_size:
            ws.destroy()
        else:
            self.disp_ques()
            self.disp_opt()

    def buttons(self):
        next_button = Button(
            ws,
            text="Next",
            command=self.next_btn,
            width=10,
            bg="#F2780C",
            fg="white",
            font=("ariel", 16, "bold")
        )

        next_button.place(x=350, y=380)

        quit_button = Button(
            ws,
            text="Quit",
            command=ws.destroy,
            width=5,
            bg="black",
            fg="white",
            font=("ariel", 16, " bold")
        )

        quit_button.place(x=700, y=50)

    def disp_opt(self):
        val = 0
        self.opt_sel.set(0)

        for option in options[self.qno]:
            self.opts[val]['text'] = option
            val += 1

    def disp_ques(self):
        qno = Label(
            ws,
            text=question[self.qno],
            width=60,
            font=('ariel', 16),
            anchor='w',
            wraplength=700,
            justify='center'
        )

        qno.place(x=70, y=100)

    def disp_title(self):
        title = Label(
            ws,
            text="OSU GUI QA",
            width=50,
            bg="#F2A30F",
            fg="white",
            font=("ariel", 20, "bold")
        )

        title.place(x=0, y=2)

    def radio_buttons(self):
        q_list = []

        y_pos = 150

        while len(q_list) < 4:

            radio_btn = Radiobutton(
                ws,
                text=" ",
                variable=self.opt_sel,
                value=len(q_list)+1,
                font=("ariel", 14)
            )
            q_list.append(radio_btn)

            radio_btn.place(x=100, y=y_pos)

            y_pos += 40

        return q_list

ws = Tk()

ws.geometry("800x450")

ws.title("OSU GUI QA")

with open('data.json') as f:
    data = json.load(f)

question = (data['question'])
options = (data['options'])
answer = (data['answer'])

quiz = Quiz()

ws.protocol("WM_DELETE_WINDOW", ws.destroy)
ws.mainloop()
