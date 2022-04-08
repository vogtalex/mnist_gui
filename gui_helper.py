from ast import Global
from pyexpat import model
from models.net import *
from csv_gui import *
from updated_tsne import *
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
                      
#For custom datasets:

# npys = './npys'
# eps = 'e1'
# examples = 'examples'
# limit = 10000

# images_orig = np.load(os.path.join(npys, 'traindata.npy')
#                       ).astype(np.float64)[:limit]

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
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()