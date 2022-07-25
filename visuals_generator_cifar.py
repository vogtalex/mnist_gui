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

with open('config.json') as f:
   config = json.load(f)

if config['Cifar']['enabled'] == True:
    npys = config['Cifar']['weightDir']
else:
    print("Failed to load cifar data")
#what epsilon for embedding
eps = config['Histogram']['weightDir']
#what attack level of example point
exeps = config['Histogram']['weightDir']
test = 'test'
examples = 'examples'
eps_dict = {'e0':'Epsilon 0', 'e1':'Epsilon 2', 'e2': 'Epsilon 4', 'e3':'Epsilon 6', 'e4':'Epsilon 10'}

cifar_dict =    {0 : "airplane",
                1: "automobile",
                2: "bird",
                3: "cat",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck"}


limit = 9000

images_orig = np.load(os.path.join(npys, examples, eps, 'advdata.npy')
                      ).astype(np.float64)[:limit]

images = []
for i in range(len(images_orig)):
    temp = images_orig[i].reshape(3, 32, 32)
    temp = temp.swapaxes(0,1)
    temp = temp.swapaxes(1,2)
    images.append(temp)

images_orig_unattacked = np.load(os.path.join(npys, examples, 'e0', 'advdata.npy')
                      ).astype(np.float64)[:limit]

images_unattacked = []
for i in range(len(images_orig)):
    temp = images_orig_unattacked[i].reshape(3,32,32)
    temp = temp.swapaxes(0,1)
    temp = temp.swapaxes(1,2)
    images_unattacked.append(temp)


# Generates an unlabeled image
def generateUnlabeledImage(count):
    imageTitle = "What is this image?"
    image = images[count]
    plt.title(imageTitle)
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()

# Generates an unlabeled image
def generateUnattackedImage(count):
    imageTitle = "What is this image?"
    image = images_unattacked[count]
    plt.title(imageTitle)
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()

def get_data(npys,eps,examples,exeps):
        #train data
        # trainlabels = np.load(os.path.join(npys,'trainlabels.npy')).astype(np.float64)[:limit]
        # trainoutput = np.load(os.path.join(npys,'trainoutput.npy')).astype(np.float64)[:limit]
        # traindata = np.load(os.path.join(npys,'traindata.npy')).astype(np.float64)[:limit]

        #adversarial data
        testlabels = np.load(os.path.join(npys,test, eps,'testlabels.npy'))[:limit]
        advoutput = np.load(os.path.join(npys,test,eps,'advoutput.npy')).astype(np.float64)[:limit]
        advdata = np.load(os.path.join(npys,test,eps,'advdata.npy')).astype(np.float64)[:limit]

        #example data
        exlabels = np.load(os.path.join(npys,examples,exeps,'testlabels.npy'))[:limit]
        exoutput = np.load(os.path.join(npys,examples,exeps,'advoutput.npy')).astype(np.float64)[:limit]
        exdata = np.load(os.path.join(npys,examples,exeps,'advdata.npy')).astype(np.float64)[:limit]
        # return trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata
        return testlabels, advoutput, advdata, exlabels, exoutput, exdata


def findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx):
        k=10
        # print("Index: ",idx)
        example = exdata[idx]
        label = np.argmax(exoutput[idx])
        # print("Model prediction: ", label)

        l = advdata - example

        norms = np.linalg.norm(l, axis=1)
        #norms = np.linalg.norm(l,ord=np.inf, axis=1)


        top = np.argpartition(norms,k-1)

        # print("True label: ", int(exlabels[idx]))
        #print("Nearest 10 labels: ")
        #print(top[:k])
        # print([(int(testlabels[i])) for i in top[:k]])
        #print("Distance to nearest 10 points: ")
        # print([(norms[idx]) for idx in top[1:k]])
        return norms, top[1:k],label,int(exlabels[idx])


def labelAxes(axs, plt):
    count = 0
    label = str(count)
    labels =    ["airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",]
    for ax in axs:
        ax.set_label(None)
        ax.set_title(None)
        ax.set_xlabel(None)
        ax.set(xticklabels=[])
        ax.set_title(labels[count], fontstyle='italic', x = 0.8, y = 0.3)
        ax.get_xaxis().set_visible(False)
        count += 1
        label = str(count)
        ax.set_ylim([0, 0.25])
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)


def generateHistograms(idx, plotID):

    ###HISTOGRAMS###################
    #idx 10 = all epsilon histogram

    b=None
    r=None
    #r=(0.99,1.01)
    r=(5,40)
    b=200

    maxHeight = 0

    fig, axs = plt.subplots(10)
    figE0, axsE0 = plt.subplots(10)
    figE2, axsE2 = plt.subplots(10)
    figE4, axsE4 = plt.subplots(10)
    figE6, axsE6 = plt.subplots(10)
    figE8, axsE8 = plt.subplots(10)

    """EPSILON 0"""
    if (plotID == 0 or plotID == 10):
        testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e0',examples,exeps)
        norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

        for i in range(10):
            y, _, _ = axsE0[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 0', histtype="step")
            axsE0[i].set_ylim([0, 1])
            if i == 0:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='unattacked', histtype="step")
            else:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, histtype="step")
            axs[i].text(13,.25,str(i),ha='center')
            axs[i].set_ylim([0, 1])
            currMax = y.max()
            if (maxHeight < currMax):
                maxHeight = currMax



    """EPSILON 2"""

    if (plotID == 2 or plotID == 10):
        testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e1',examples,exeps)
        norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

        for i in range(10):
            y, _, _ = axsE2[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 0.5', histtype="step")
            axsE2[i].set_ylim([0, 1])
            if i == 0:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 0.5', histtype="step")
            else:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, histtype="step")
            axs[i].set_ylim([0, 1])

            currMax = y.max()
            if (maxHeight < currMax):
                maxHeight = currMax



    ##################
    """EPSILON 4"""

    if (plotID == 4 or plotID == 10):
        testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e2',examples,exeps)
        norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

        for i in range(10):
            y, _, _ = axsE4[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 1', histtype="step")
            axsE4[i].set_ylim([0, 1])
            if i == 0:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 1', histtype="step")
            else:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, histtype="step")
            axs[i].set_ylim([0, 1])

            currMax = y.max()
            if (maxHeight < currMax):
                maxHeight = currMax


    ###########
    """EPSILON 6"""

    if (plotID == 6 or plotID == 10):
        testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e3',examples,exeps)
        norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

        for i in range(10):
            y, _, _ = axsE6[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 1.5', histtype="step")
            axsE6[i].set_ylim([0, 1])
            if i == 0:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 1.5', histtype="step")
            else:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, histtype="step")
            axs[i].set_ylim([0, 1])
            currMax = y.max()
            if (maxHeight < currMax):
                maxHeight = currMax


    """EPSILON 8"""
    if (plotID == 8 or plotID == 10):
        testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e4',examples,exeps)
        norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

        for i in range(10):
            y, _, _ = axsE8[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 2', histtype="step")
            axsE8[i].set_ylim([0, 1])
            if i == 0:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 2', histtype="step")
            else:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, histtype="step")
            axs[i].set_ylim([0, 1])
            currMax = y.max()
            if (maxHeight < currMax):
                maxHeight = currMax


    # title = "Model Prediction: %d" % (prediction)

    # plt.suptitle(title)
    plt.legend(loc='upper left')

    if plotID == 10:
        fig.suptitle("All Epsilons")
        fig.legend(loc='upper left')
        labelAxes(axs, fig)
        return(fig)
    if plotID == 0:
        figE0.suptitle("Epsilon 0")
        labelAxes(axsE0, figE0)
        return(figE0, maxHeight)
    if plotID == 2:
        figE2.suptitle("Epsilon 0.5")
        labelAxes(axsE2, figE2)
        return(figE2, maxHeight)
    if plotID == 4:
        figE4.suptitle("Epsilon 1")
        labelAxes(axsE4, figE4)
        return(figE4, maxHeight)
    if plotID == 6:
        figE6.suptitle("Epsilon 1.5")
        labelAxes(axsE6, figE6)
        return(figE6, maxHeight)
    if plotID == 8:
        figE8.suptitle("Epsilon 2")
        labelAxes(axsE8, figE8)
        return(figE8, maxHeight)

def generateBoxPlot(idx):
    b=None
    r=None
    #r=(0.99,1.01)
    r=(5,16)
    b=200
    fig, axs = plt.subplots()
    norm_list = []

    testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e0',examples,exeps)
    norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)
    norm_list.append(norms)

    testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e1',examples,exeps)
    norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)
    norm_list.append(norms)

    testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e2',examples,exeps)
    norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)
    norm_list.append(norms)

    testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e3',examples,exeps)
    norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)
    norm_list.append(norms)

    testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e4',examples,exeps)
    norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)
    norm_list.append(norms)

    print(cifar_dict[truelabel])
    actual_prediction = cifar_dict[prediction]
    title = "Model Prediction: %s" % (actual_prediction)
    plt.suptitle(title)
    axs.boxplot(norm_list, patch_artist = True,notch ='True', vert = 1,labels=['unattacked','Epsilon 0.5', 'Epsilon 1', 'Epsilon 1.5', 'Epsilon 2'], showmeans=True)
    return fig
