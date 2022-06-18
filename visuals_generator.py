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

if config['MNIST']['enabled'] == True:
    npys = config['MNIST']['weightDir']
else:
    print("Failed to load mnist data")

#what epsilon for embedding
eps = config['Histogram']['weightDir']
#what attack level of example point
exeps = config['Histogram']['weightDir']
test = 'test'
examples = 'examples'
eps_dict = {'e0':'Epsilon 0.0', 'e1':'Epsilon 2', 'e2': 'Epsilon 4', 'e3':'Epsilon 6', 'e4':'Epsilon 10'}


limit = 9000

images_orig = np.load(os.path.join(npys, examples, eps, 'advdata.npy')
                      ).astype(np.float64)[:limit]

images = []
for i in range(len(images_orig)):
    images.append(images_orig[i].reshape(28, 28))

images_orig_unattacked = np.load(os.path.join(npys, examples, 'e0', 'advdata.npy')
                      ).astype(np.float64)[:limit]

images_unattacked = []
for i in range(len(images_orig)):
    images_unattacked.append(images_orig_unattacked[i].reshape(28, 28))

# Generates an unlabeled image
def generateUnlabeledImage(count):
    imageTitle = "What is this number?"
    image = images[count]
    plt.title(imageTitle)
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()

# Generates an unlabeled image
def generateUnattackedImage(count):
    imageTitle = "What is this number?"
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
        testlabels = np.load(os.path.join(npys, eps,'testlabels.npy')).astype(np.float64)[:limit]
        advoutput = np.load(os.path.join(npys,eps,'advoutput.npy')).astype(np.float64)[:limit]
        advdata = np.load(os.path.join(npys,eps,'advdata.npy')).astype(np.float64)[:limit]
        
        #example data
        exlabels = np.load(os.path.join(npys,examples,exeps,'testlabels.npy')).astype(np.float64)[:limit]
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
    for ax in axs:
        ax.set_title(label, fontstyle='italic', x = 0.8, y = 0.3)
        ax.get_xaxis().set_visible(False)
        count += 1
        label = str(count)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

def generateTSNEPlots(idx):

    plt.clf()
    # trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,eps,examples,exeps)
    testlabels, advoutput, origdata, exlabels, exoutput, exdata = get_data(npys,'e0',examples,exeps)
    testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,eps,examples,exeps)

    #print("advdata ",advdata.shape)
    #print("exdata ",exdata.shape)
    #print("testlabels ",testlabels.shape)
    #print("advoutput ",advoutput.shape)

    #norms,idxs,prediction,truelabel = findNearest()
    norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

    exdata1 = np.load(os.path.join(npys,examples,'e0','advdata.npy')).astype(np.float64)[:limit]
    img = exdata1[idx].reshape((28,28))


    exdata2 = np.load(os.path.join(npys,examples,exeps,'advdata.npy')).astype(np.float64)[:limit]
    img = exdata2[idx].reshape((28,28))

    print(np.linalg.norm(exdata1[idx] - exdata2[idx]))

    print('max distance', max(norms))
    print('min distance', min(norms))
    print('avg distance', sum(norms)/len(norms))
    #print('data shape: ', traindata.shape)
    #print('labels shape: ', trainlabels.shape)
    #print('output shape: ', trainoutput.shape)



    #for combining data/advdata
    #data = np.append(data, advdata, axis=0)

    X_2d = []
    if exists("embedding.npy"):
        X_2d = np.load('embedding.npy').astype(np.float64)
    else:
        tsne = TSNE(n_components=2, random_state=4,perplexity=100)
        X_2d = tsne.fit_transform(origdata)
        np.save('./embedding.npy', X_2d, allow_pickle=False)


    labels = list(range(0, 10))
    target_ids = range(10)
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'
    
    fig, (ax1,ax2) = plt.subplots(1,2)

    #plot embedding for class coloration
    for i, c, label in zip(target_ids, colors, labels):
        ax1.scatter(X_2d[(testlabels[...] == i), 0],
                X_2d[(testlabels[...] == i), 1],
                c=c,
                label=label,
                s=3,
                picker=True)

    ax1.set_title("Test Data")

    #plot embedding for norm coloration
    ax2.scatter(X_2d[..., 0],
            X_2d[..., 1],
            c=norms[...],
            s=3,
            cmap='viridis')

    #plot 10 nearest points
    cb = ax2.scatter(X_2d[idxs,0],X_2d[idxs,1],
            c='red',
            label="nearest",
            s=10,
            picker=True)

    title = "Model Prediction: %d\nActual Label: %d\nAverage Distance: %f" % (prediction,truelabel,float(sum(norms))/len(norms))
    ax2.set_title(title)

    plt.colorbar(cb,label="norm")
    cb.set_clim(5,15)
    
    ax1.legend()
    # plt.show()
    return fig


def generateHistograms(idx, plotID):

    ###HISTOGRAMS###################
    #idx 10 = all epsilon histogram

    b=None
    r=None
    #r=(0.99,1.01)
    r=(5,16)
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
            y, _, _ = axsE2[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 2', histtype="step")
            axsE2[i].set_ylim([0, 1])
            if i == 0:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 2', histtype="step")
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
            y, _, _ = axsE4[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 4', histtype="step")
            axsE4[i].set_ylim([0, 1])
            if i == 0:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 4', histtype="step")
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
            y, _, _ = axsE6[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 6', histtype="step")
            axsE6[i].set_ylim([0, 1])
            if i == 0:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 6', histtype="step")
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
            y, _, _ = axsE8[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 8', histtype="step")
            axsE8[i].set_ylim([0, 1])
            if i == 0:
                axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 8', histtype="step")
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
        return(fig)
    if plotID == 0:
        figE0.suptitle("Epsilon 0")
        labelAxes(axsE0, figE0)
        return(figE0, maxHeight)
    if plotID == 2:
        figE2.suptitle("Epsilon 2")
        labelAxes(axsE2, figE2)
        return(figE2, maxHeight)
    if plotID == 4:
        figE4.suptitle("Epsilon 4")
        labelAxes(axsE4, figE4)
        return(figE4, maxHeight)
    if plotID == 6:
        figE6.suptitle("Epsilon 6")
        labelAxes(axsE6, figE6)
        return(figE6, maxHeight)
    if plotID == 8:
        figE8.suptitle("Epsilon 8")
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

    title = "Model Prediction: %d" % (prediction)
    plt.suptitle(title)
    axs.boxplot(norm_list, patch_artist = True,notch ='True', vert = 1,labels=['unattacked','Epsilon 2', 'Epsilon 4', 'Epsilon 6', 'Epsilon 8'], showmeans=True)
    return fig