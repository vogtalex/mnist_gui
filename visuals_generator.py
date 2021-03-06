# Load Python Libraries
import numpy as np
import os
import gzip, pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.manifold import TSNE
import torch
import json
import math

with open('config.json') as f:
   config = json.load(f)

if config['MNIST']['enabled'] == True:
    npys = config['MNIST']['weightDir']
else:
    print("Failed to load mnist data")

#what epsilon for embedding
eps = config['Histogram']['weightDir']
#what attack level of example point
exeps = config['Histogram']['weightDir']
examples = 'examples'
# eps_dict = {'e0':'Epsilon 0', 'e1':'Epsilon 2', 'e2': 'Epsilon 4', 'e3':'Epsilon 6', 'e4':'Epsilon 10'}

# only use this # of images from the image set(s)
limit = 9000

images_orig = np.load(os.path.join(npys, examples, eps, 'advdata.npy')).astype(np.float64)[:limit]
images = []
for i in range(len(images_orig)):
    images.append(images_orig[i].reshape(28, 28))

images_orig_unattacked = np.load(os.path.join(npys, examples, 'e0', 'advdata.npy')).astype(np.float64)[:limit]
images_unattacked = []
for i in range(len(images_orig)):
    images_unattacked.append(images_orig_unattacked[i].reshape(28, 28))

# Generates an unlabeled image
def generateUnlabeledImage(count):
    image = images[count]
    #imageTitle = "What is this number?"
    #plt.title(imageTitle)
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()

# Generates an unlabeled image
def generateUnattackedImage(count):
    image = images_unattacked[count]
    #imageTitle = "What is this number?"
    #plt.title(imageTitle)
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()

def cached_get_data():
    cache = {}
    def __get_data(npys,eps,examples,exeps):
        if eps not in cache:
            #adversarial data
            testlabels = np.load(os.path.join(npys, eps,'testlabels.npy')).astype(np.float64)[:limit]
            advoutput = np.load(os.path.join(npys,eps,'advoutput.npy')).astype(np.float64)[:limit]
            advdata = np.load(os.path.join(npys,eps,'advdata.npy')).astype(np.float64)[:limit]

            #example data
            exlabels = np.load(os.path.join(npys,examples,exeps,'testlabels.npy')).astype(np.float64)[:limit]
            exoutput = np.load(os.path.join(npys,examples,exeps,'advoutput.npy')).astype(np.float64)[:limit]
            exdata = np.load(os.path.join(npys,examples,exeps,'advdata.npy')).astype(np.float64)[:limit]

            cache[eps] = [testlabels,advoutput,advdata,exlabels,exoutput,exdata]
            # return trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata
        return tuple(cache[eps])

    return __get_data
get_data = cached_get_data()

def findNearest(exdata,exoutput,exlabels,advdata, _,idx):
    k=10
    example = exdata[idx]
    label = np.argmax(exoutput[idx])

    norms = np.linalg.norm(advdata - example, axis=1)

    top = np.argpartition(norms, k-1)
    return norms, top[1:k], label, int(exlabels[idx])

def labelAxes(axs, plt):
    count = 0
    label = str(count)
    for ax in axs:
        ax.set_title(label, fontstyle='italic', x = 0.8, y = 0.0)
        ax.get_xaxis().set_visible(False)
        count += 1
        label = str(count)
    axs[len(axs)-1].get_xaxis().set_visible(True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

def generateTSNEPlots(idx):
    # plt.clf()
    testlabels, advoutput, origdata, exlabels, exoutput, exdata = get_data(npys,'e0',examples,exeps)
    testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,eps,examples,exeps)

    norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

    print('max distance', max(norms))
    print('min distance', min(norms))
    print('avg distance', sum(norms)/len(norms))

    X_2d = []
    if exists("embedding.npy"):
        X_2d = np.load('embedding.npy').astype(np.float64)
    else:
        tsne = TSNE(n_components=2, random_state=4, perplexity=100)
        X_2d = tsne.fit_transform(origdata)
        np.save('./embedding.npy', X_2d, allow_pickle=False)

    labels = list(range(0, 10))
    target_ids = range(10)
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'

    fig, (ax1,ax2) = plt.subplots(1,2)

    #plot embedding for class coloration
    for i, c, label in zip(target_ids, colors, labels):
        ax1.scatter(X_2d[(testlabels[...] == i), 0], X_2d[(testlabels[...] == i), 1], c=c, label=label, s=3, picker=True)

    ax1.set_title("Test Data")

    #plot embedding for norm coloration
    ax2.scatter(X_2d[..., 0], X_2d[..., 1], c=norms[...], s=3, cmap='viridis')

    #plot 10 nearest points
    cb = ax2.scatter(X_2d[idxs,0],X_2d[idxs,1], c='red', label="nearest", s=10, picker=True)

    title = f"Model Prediction: {prediction}\nActual Label: {truelabel}\nAverage Distance: {float(sum(norms))/len(norms)}"
    ax2.set_title(title)

    plt.colorbar(cb,label="norm")
    cb.set_clim(5,15)

    ax1.legend()
    return fig

def roundSigFigs(num, sigFigs):
    return str(num)[:(int(math.log(num,10)) + 2 + sigFigs if num else 1)]

maxEpsilon = config["General"]["maxEpsilon"]
epsilonStepSize = config["General"]["epsilonStepSize"]
# finds # of significant figures after the decimal place of the step size
sigFigs = len(repr(float(epsilonStepSize)).split('.')[1].rstrip('0'))
epsilonList = [x * epsilonStepSize for x in range(0, math.ceil(maxEpsilon*(1/epsilonStepSize)))]
def generateHistograms(idx, plotID):
    r=(5,16)
    b=200

    maxHeight = 0
    fig, axs = plt.subplots(10)

    if plotID == maxEpsilon:
        for epsilon in epsilonList:
            testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,f'e{roundSigFigs(epsilon,sigFigs)}',examples,exeps)
            norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

            for i in range(10):
                if i:
                    y, _, _ = axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, histtype="step")
                else:
                    y, _, _ = axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label=f"Epsilon {epsilon}", histtype="step")
                axs[i].set_ylim([0, 1])
                maxHeight = max(maxHeight,y.max())
        fig.legend(loc='upper left')
        fig.suptitle("All Epsilons")
    else:
        testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,f'e{roundSigFigs(plotID,sigFigs)}',examples,exeps)
        norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels,idx)

        for i in range(10):
            y, _, _ = axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label=f"Epsilon {plotID}", histtype="step")
            axs[i].set_ylim([0, 1])
            maxHeight = max(maxHeight,y.max())
        fig.suptitle(f"Epsilon {plotID}")

    labelAxes(axs, fig)
    return(fig, maxHeight)

def generateBoxPlot(idx):
    fig, axs = plt.subplots()
    norm_list = []

    for epsilon in epsilonList:
        testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys, f'e{roundSigFigs(epsilon,sigFigs)}', examples, exeps)
        norms,idxs,prediction,truelabel = findNearest(exdata, exoutput, exlabels, advdata, testlabels, idx)
        norm_list.append(norms)

    plt.suptitle(f"Model Prediction: {prediction}")
    axs.boxplot(norm_list, patch_artist=True, notch='True', vert=1, labels=[f"Epsilon {epsilon}" for epsilon in epsilonList], showmeans=True)
    return fig
