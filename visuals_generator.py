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

try:
    npys = config['MNIST']['weightDir']
except:
    exit("Data path not valid")

#what epsilon for embedding
displayEpsilon = f"e{config['General']['displayEpsilon']}"
examples = 'examples'

# only use this # of images from the image set(s)
limit = 9000

#example data
exlabels = np.load(os.path.join(npys,examples,displayEpsilon,'testlabels.npy')).astype(np.float64)[:limit]
exoutput = np.load(os.path.join(npys,examples,displayEpsilon,'advoutput.npy')).astype(np.float64)[:limit]
exdata = np.load(os.path.join(npys,examples,displayEpsilon,'advdata.npy')).astype(np.float64)[:limit]

images = [image.reshape(28, 28) for image in np.load(os.path.join(npys, examples, displayEpsilon, 'advdata.npy')).astype(np.float64)[:limit]]
images_unattacked = [image.reshape(28, 28) for image in np.load(os.path.join(npys, examples, 'e0', 'advdata.npy')).astype(np.float64)[:limit]]

# Generates an unlabeled image
def generateUnlabeledImage(count):
    image = images[count]
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()

# Generates an unattacked image
def generateUnattackedImage(count):
    image = images_unattacked[count]
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()

def cached_get_data():
    cache = {}
    def __get_data(npys,eps):
        if eps not in cache:
            #adversarial data
            testlabels = np.load(os.path.join(npys, eps,'testlabels.npy')).astype(np.float64)[:limit]
            # advoutput = np.load(os.path.join(npys,eps,'advoutput.npy')).astype(np.float64)[:limit]
            advdata = np.load(os.path.join(npys,eps,'advdata.npy')).astype(np.float64)[:limit]

            cache[eps] = [testlabels,advdata]
        return tuple(cache[eps])

    return __get_data
get_data = cached_get_data()

def findNearest(exdata,exoutput,exlabels,advdata,_,idx):
    k=10
    example = exdata[idx]
    label = np.argmax(exoutput[idx])

    norms = np.linalg.norm(advdata - example, axis=1)

    top = np.argpartition(norms, k-1)
    # returns norms of all data, the nearest k points, the predicted label, and the actual label
    return norms, top[1:k], label, int(exlabels[idx])

def labelAxes(axs, plt):
    count = 0
    for ax in axs:
        ax.set_title(str(count), fontstyle='italic', x = 0.8, y = 0.0)
        ax.get_xaxis().set_visible(False)
        count += 1
    axs[count-1].get_xaxis().set_visible(True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

def generateTSNEPlots(idx):
    testlabels, origdata = get_data(npys,'e0')
    testlabels, advdata = get_data(npys,displayEpsilon)

    norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

    print('max distance', max(norms))
    print('min distance', min(norms))
    print('avg distance', sum(norms)/len(norms))

    X_2d = []
    if os.path.exists("./embedding.npy"):
        X_2d = np.load('./embedding.npy').astype(np.float64)
    else:
        tsne = TSNE(n_components=2, random_state=4, perplexity=100)
        X_2d = tsne.fit_transform(origdata)
        np.save('./embedding.npy', X_2d, allow_pickle=False)

    labels = list(range(10))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'

    fig, (ax1,ax2) = plt.subplots(1,2)

    #plot embedding for class coloration
    for c, label in zip(colors, labels):
        ax1.scatter(X_2d[(testlabels[...] == label), 0], X_2d[(testlabels[...] == label), 1], c=c, label=label, s=3, picker=True)

    ax1.set_title("Test Data")

    #plot embedding for norm coloration
    ax2.scatter(X_2d[..., 0], X_2d[..., 1], c=norms[...], s=3, cmap='viridis')

    #plot 10 nearest points
    cb = ax2.scatter(X_2d[idxs,0],X_2d[idxs,1], c='red', label="nearest", s=10, picker=True)

    ax2.set_title(f"Model Prediction: {prediction}\nAverage Distance: {round(float(sum(norms))/len(norms),2)}")

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
            testlabels, advdata = get_data(npys,f'e{roundSigFigs(epsilon,sigFigs)}')
            norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

            for i in range(10):
                if i:
                    y, _, _ = axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, histtype="step")
                else:
                    y, _, _ = axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label=f"Epsilon {epsilon}", histtype="step")
                maxHeight = max(maxHeight,y.max())
        fig.legend(loc='upper left')
        fig.suptitle("All Epsilons")
    else:
        testlabels, advdata = get_data(npys,f'e{roundSigFigs(plotID,sigFigs)}')
        norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels,idx)

        for i in range(10):
            y, _, _ = axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label=f"Epsilon {plotID}", histtype="step")
            maxHeight = max(maxHeight,y.max())
        fig.suptitle(f"Epsilon {plotID}")

    # should it be 0-1 or 0-max?
    for ax in axs:
        # ax.set_ylim([0, maxHeight])
        ax.set_ylim([0, 1])

    labelAxes(axs, fig)
    return(fig, maxHeight)

def generateBoxPlot(idx):
    fig, axs = plt.subplots()
    norm_list = []

    for epsilon in epsilonList:
        testlabels, advdata = get_data(npys, f'e{roundSigFigs(epsilon,sigFigs)}')
        norms,idxs,prediction,truelabel = findNearest(exdata, exoutput, exlabels, advdata, testlabels, idx)
        norm_list.append(norms)

    plt.suptitle(f"Model Prediction: {prediction}")
    axs.boxplot(norm_list, patch_artist=True, notch='True', vert=1, labels=[f"Epsilon {epsilon}" for epsilon in epsilonList], showmeans=True)
    return fig
