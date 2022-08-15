# Load Python Libraries
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import math
from functions import generateEpsilonList
import torch
import torch.nn as nn
import torch.optim
from collections import OrderedDict
from matplotlib.pyplot import subplot
from chained_AE import Autoencoder, Chained_AE
from mnist_cost import MNISTCost

with open('config.json') as f:
   config = json.load(f)

try:
    npys = config['Model']['outputDir']
except:
    exit("Data path not valid")

#what epsilon for embedding
displayEpsilon = f"e{config['General']['displayEpsilon']}"
examples = 'examples'

# only use this # of images from the image set(s)
limit = 9000

# constants for trajectory regression
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dist_metric = 'l2'
attack_type = 'targeted'
d=50
max_epsilon = 6
batch_size = 10 # MUST NOT BE LOWER THAN 10

#example data
exlabels = np.load(os.path.join(npys,examples,displayEpsilon,'testlabels.npy')).astype(np.float64)[:limit]
exoutput = np.load(os.path.join(npys,examples,displayEpsilon,'advoutput.npy')).astype(np.float64)[:limit]
exdata = np.load(os.path.join(npys,examples,displayEpsilon,'advdata.npy')).astype(np.float64)[:limit]

testlabels = np.load(os.path.join(npys, 'e0','testlabels.npy')).astype(np.float64)[:limit]

# this is kinda a makeshift solution, do it better later
labels = list(set(exlabels))

imageData = np.load(os.path.join(npys, examples, displayEpsilon, 'advdata.npy')).astype(np.float64)[:limit]
images = imageData.reshape(imageData.shape[:-1]+(28,28))
images_unattacked = [image.reshape(28, 28) for image in np.load(os.path.join(npys, examples, 'e0', 'advdata.npy')).astype(np.float64)[:limit]]

# Generates an unlabeled image
def generateUnlabeledImage(idx):
    image = images[idx]
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()

# Generates an unattacked image
def generateUnattackedImage(idx):
    image = images_unattacked[idx]
    plt.figure(figsize=(4,3))
    plt.imshow(image, cmap="gray")
    return plt.gcf()

def cached_get_data():
    cache = {}
    def __get_data(npys,eps):
        if eps not in cache:
            #adversarial data
            # testlabels = np.load(os.path.join(npys, eps,'testlabels.npy')).astype(np.float64)[:limit]
            # advoutput = np.load(os.path.join(npys,eps,'advoutput.npy')).astype(np.float64)[:limit]
            advdata = np.load(os.path.join(npys,eps,'advdata.npy')).astype(np.float64)[:limit]

            cache[eps] = advdata
        return cache[eps]

    return __get_data
get_data = cached_get_data()

def getTrueLabel(idx):
    return exlabels[idx]

def findNearest(exdata,exoutput,exlabels,advdata,_,idx):
    k=10
    example = exdata[idx]
    label = np.argmax(exoutput[idx])

    norms = np.linalg.norm(advdata - example, axis=1)

    top = np.argpartition(norms, k-1)
    # returns norms of all data, the nearest k points, the predicted label, and the actual label
    return norms, top[1:k], label, exlabels[idx]

def labelAxes(axs, plt):
    count = 0
    for ax in axs:
        ax.set_title(labels[count], fontstyle='italic', x = 0.8, y = 0.0)
        ax.get_xaxis().set_visible(False)
        count += 1
    axs[count-1].get_xaxis().set_visible(True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

def generateTSNEPlots(idx):
    origdata = get_data(npys,'e0')
    advdata = get_data(npys,displayEpsilon)

    norms,idxs,prediction,_ = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

    # print('max distance', max(norms))
    # print('min distance', min(norms))
    # print('avg distance', sum(norms)/len(norms))

    X_2d = []
    if os.path.exists("./embedding.npy"):
        X_2d = np.load('./embedding.npy').astype(np.float64)
    else:
        tsne = TSNE(n_components=2, random_state=4, perplexity=100)
        X_2d = tsne.fit_transform(origdata)
        np.save('./embedding.npy', X_2d, allow_pickle=False)

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
    # cb.set_clim(min(norms),max(norms))

    ax1.legend()
    return fig

def roundSigFigs(num, sigFigs):
    return str(num)[:(int(math.log(num,10))*(num>1) + 2 + sigFigs if num else 1)]

maxEpsilon = config["General"]["maxEpsilon"]
epsilonStepSize = config["General"]["epsilonStepSize"]
# finds # of significant figures after the decimal place of the step size
sigFigs = len(repr(float(epsilonStepSize)).split('.')[1].rstrip('0'))
epsilonList = generateEpsilonList(epsilonStepSize,maxEpsilon)
def generateHistograms(idx, plotID, height = None):
    r=(5,16)
    b=200

    maxHeight = 0
    fig, axs = plt.subplots(10)

    if plotID > maxEpsilon:
        for epsilon in epsilonList:
            advdata = get_data(npys,f'e{roundSigFigs(epsilon,sigFigs)}')
            norms,_,_,_ = findNearest(exdata,exoutput,exlabels,advdata,testlabels, idx)

            for i in range(10):
                arr = norms[(testlabels[...] == labels[i])]
                weights = np.ones_like(arr)/len(arr)
                if i:
                    y, _, _ = axs[i].hist(arr, weights=weights, alpha=0.5, bins=b,range=r,density=False, histtype="step")
                else:
                    y, _, _ = axs[i].hist(arr, weights=weights, alpha=0.5, bins=b,range=r,density=False, histtype="step", label=f"Epsilon {epsilon}")
                maxHeight = max(maxHeight,y.max())
        fig.legend(loc='upper left')
        fig.suptitle("All Epsilons")
    else:
        advdata = get_data(npys,f'e{roundSigFigs(plotID,sigFigs)}')
        norms,_,_,_ = findNearest(exdata,exoutput,exlabels,advdata,testlabels,idx)
        normSum = sum(norms)
        for i in range(10):
            arr = norms[(testlabels[...] == labels[i])]
            weights = np.ones_like(arr)/len(arr)
            y, _, _ = axs[i].hist(arr, weights=weights, alpha=0.5, bins=b,range=r,density=False, label=f"Epsilon {plotID}", histtype="step")
            maxHeight = max(maxHeight,y.max())
        fig.suptitle(f"Epsilon {plotID}")

    # should it be 0-1 or 0-max?
    for ax in axs:
        ax.set_ylim([0, height if height else maxHeight])

    labelAxes(axs, fig)
    return(fig, maxHeight)

def generateBoxPlot(idx):
    fig, axs = plt.subplots()
    norm_list = []

    for epsilon in epsilonList:
        advdata = get_data(npys, f'e{roundSigFigs(epsilon,sigFigs)}')
        norms,_,prediction,_ = findNearest(exdata, exoutput, exlabels, advdata, testlabels, idx)
        norm_list.append(norms)

    plt.suptitle(f"Model Prediction: {prediction}")
    axs.boxplot(norm_list, patch_artist=True, notch='True', vert=1, labels=[f"Epsilon {epsilon}" for epsilon in epsilonList], showmeans=True)
    return fig

# function for loading a segment of the autoencoders chain
def load_part_of_model(model, state_dict, numblocks):
    block_diff = 6-numblocks
    if block_diff != 0:
        new_state_dict = OrderedDict()
        keyslist = list(state_dict.keys())
        keys_to_rmv = []
        for i in range(block_diff):
            keys_to_rmv.append('ae.{}'.format(i))

        for key in keyslist:
            isbad = False

            for badkey in keys_to_rmv:
                if key.startswith(badkey):
                    isbad = True

            if not isbad:
                val = state_dict[key]
                newkey = 'ae.{}'.format(int(key[3])-block_diff) + key[4:]
                new_state_dict[newkey] = val
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    return model

# function for constructing the autoencoders
def make_models(dist_metric, epsilons, attack_type, d):
    numblocks = len(epsilons)-1
    ae = Chained_AE(block=Autoencoder, num_blocks=numblocks, codeword_dim=d, fc2_input_dim=128)
    pt_state_dict = torch.load('./model/' + attack_type + '/chained_ae_' + dist_metric +
        '_{}_to_{}_d={}_epc{}.pth'.format(6, 0, d, 500))
    ae = load_part_of_model(ae, pt_state_dict, numblocks)
    ae.to(device)
    ae.eval()

    return ae

def make_trajectory(expected, ae):
    return ae(expected, True)

# function for rounding predicted cost to the assigned bins, note that this is for l2 only
def round_cost(pc):
    rounded_cost = np.clip(pc, 0, max_epsilon)
    rounded_cost = np.round(rounded_cost)
    return rounded_cost

def buildTrajectoryCostReg(idx):
    def __trajectoryCostReg(idx):
        # load cost regression model
        cost_reg = MNISTCost()
        cost_reg.load_state_dict(torch.load('./model/costreg_mnist_l2_ce.pth'))
        cost_reg.to(device)
        # train mode is required for some strange reason, cost regression model does not work properly under eval mode
        cost_reg.train()

        localIdx = (idx - __trajectoryCostReg.startIdx) % batch_size
        batchNum = (idx - __trajectoryCostReg.startIdx) // batch_size

        # if new index requires new batch, cut out new batch and generate cost regressions
        if not localIdx:
             temp = images[__trajectoryCostReg.startIdx + batchNum*batch_size:__trajectoryCostReg.startIdx + batchNum*batch_size + batch_size]
             __trajectoryCostReg.batchData = torch.unsqueeze(torch.from_numpy(temp),1).to(torch.float)
             __trajectoryCostReg.pc = cost_reg(__trajectoryCostReg.batchData).detach().cpu().numpy()
             __trajectoryCostReg.rounded_pc = round_cost(__trajectoryCostReg.pc)

        exp = torch.unsqueeze(__trajectoryCostReg.batchData[localIdx], 0)
        cost = __trajectoryCostReg.rounded_pc[localIdx]

        reg_epsilons = range(int(cost)+1)
        reg_ae = make_models(dist_metric, reg_epsilons, attack_type, d)
        reg_recons = make_trajectory(exp, reg_ae)

        fig = plt.figure(figsize=(8,4))
        fig.suptitle('Reconstruction using estimated cost ({:.2f}), '.format(__trajectoryCostReg.pc[localIdx].item()) + f'{dist_metric} {attack_type} AE, d={d}')

        num_bins = len(reg_recons)+1
        for j in range(num_bins):
            ax = fig.add_subplot(1,num_bins,j+1,anchor='N')
            if j == len(reg_recons):
                ax.imshow(exp[0][0], cmap="gray")
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            else:
                ax.imshow(reg_recons[num_bins-2-j][0][0].detach().cpu(), cmap="gray")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title(f'Epsilon {reg_epsilons[j]}')

        plt.tight_layout()

        return fig
    __trajectoryCostReg.startIdx = idx
    __trajectoryCostReg.batchData = None
    __trajectoryCostReg.pc = 0
    __trajectoryCostReg.rounded_pc = 0
    global trajectoryCostRegFunc
    trajectoryCostRegFunc = __trajectoryCostReg

def trajectoryCostReg(idx):
    return trajectoryCostRegFunc(idx)
