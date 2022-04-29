# Load Python Libraries
import numpy as np
import os
import gzip, pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.manifold import TSNE
from functions import *
import torch
import json

from pathlib import Path
from winreg import HKEY_LOCAL_MACHINE
from csv_gui import initializeCSV, writeToCSV
from updated_tsne import generateUnlabeledImage, generateTSNEPlots

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image, ImageTk
import json



limit = 9000
#idx 41
#idx 48

with open('config.json') as f:
    config = json.load(f)

npys = config['TSNE']['weightDir']
#what epsilon for embedding
eps = config['Histogram']['weightDir']
#what attack level of example point
exeps = config['Histogram']['weightDir']
test = 'test'
examples = 'examples'
eps_dict = {'e0':'Epsilon 0.0', 'e1':'Epsilon 2', 'e2': 'Epsilon 4', 'e3':'Epsilon 6', 'e4':'Epsilon 10'}

def get_data(npys,eps,examples,exeps):
    #train data
    # trainlabels = np.load(os.path.join(npys,'trainlabels.npy')).astype(np.float64)[:limit]
    # trainoutput = np.load(os.path.join(npys,'trainoutput.npy')).astype(np.float64)[:limit]
    # traindata = np.load(os.path.join(npys,'traindata.npy')).astype(np.float64)[:limit]
    
    #adversarial data
    testlabels = np.load(os.path.join(npys,test, eps,'testlabels.npy')).astype(np.float64)[:limit]
    advoutput = np.load(os.path.join(npys,test,eps,'advoutput.npy')).astype(np.float64)[:limit]
    advdata = np.load(os.path.join(npys,test,eps,'advdata.npy')).astype(np.float64)[:limit]
    
    #example data
    exlabels = np.load(os.path.join(npys,examples,exeps,'testlabels.npy')).astype(np.float64)[:limit]
    exoutput = np.load(os.path.join(npys,examples,exeps,'advoutput.npy')).astype(np.float64)[:limit]
    exdata = np.load(os.path.join(npys,examples,exeps,'advdata.npy')).astype(np.float64)[:limit]
    # return trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata
    return testlabels, advoutput, advdata, exlabels, exoutput, exdata

# trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,eps,examples,exeps)
testlabels, advoutput, advdata3, exlabels, exoutput, exdata = get_data(npys,"e3",examples,exeps)
testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,"e0",examples,exeps)


idx = 152

exdata1 = np.load(os.path.join(npys,examples,'e0','advdata.npy')).astype(np.float64)[:limit]
img = exdata1[idx].reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

exdata2 = np.load(os.path.join(npys,examples,exeps,'advdata.npy')).astype(np.float64)[:limit]
img = exdata2[idx].reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

print(np.linalg.norm(exdata1[idx] - exdata2[idx]))
exit()


# limit = 9000

# images_orig_e3 = np.load(os.path.join(npys, examples, 'e3', 'advdata.npy')
#                       ).astype(np.float64)[:limit]

# images = []
# for i in range(len(advdata3)):
#     images.append(advdata3[i].reshape(28, 28))

# images3 = []
# for i in range(len(advdata)):
#     images3.append(advdata[i].reshape(28, 28))

# # Generates an unlabeled image
# def generateUnlabeledImage(count):
#     image = images[count]
#     image3 = images3[count]
#     f, axarr = plt.subplots(2)
#     axarr[0].imshow(image, cmap='gray')
#     axarr[1].imshow(image3, cmap='gray')
#     # plotImage = plt.imshow(image, cmap="gray")
#     plt.show()


# # diff = images3[0] - images[0]
# # print(diff)
# generateUnlabeledImage(148)

# def generateUnlabeledImage3(count):
#     image = images3[count]
#     plotImage = plt.imshow(image, cmap="gray")
#     plt.show()


# generateUnlabeledImage(0)
# generateUnlabeledImage3(0)


