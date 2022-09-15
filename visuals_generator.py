# Load Python Libraries
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
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

import time

with open('config.json') as f:
   config = json.load(f)

# get pre-generated data
try:
    npys = config['Model']['outputDir']
except:
    exit("Data path not valid")

# what epsilon for embedding
displayEpsilon = f"e{config['General']['displayEpsilon']}"
examples = 'examples'

# only use this # of images from the image set(s)
limit = 9000
# figure size downscale value
scaler = 1
# num of closest points for tsne
k=10

# constants for trajectory regression
use_cuda = False
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
dist_metric = 'l2'
attack_type = 'targeted'
d=50
max_epsilon = 6
batch_size = 10 # MUST NOT BE LOWER THAN 10

#example data
exlabels = np.load(os.path.join(npys,examples,'testlabels.npy')).astype(np.float64)[:limit]
exoutput = np.load(os.path.join(npys,examples,displayEpsilon,'advoutput.npy')).astype(np.float64)[:limit]
exdata = np.load(os.path.join(npys,examples,displayEpsilon,'advdata.npy')).astype(np.float64)[:limit]

testlabels = np.load(os.path.join(npys,'testlabels.npy')).astype(np.float64)[:limit]

# this is kinda a makeshift solution, do it better later
labels = list(map(int,set(exlabels)))

imageData = np.load(os.path.join(npys, examples, displayEpsilon, 'advdata.npy')).astype(np.float64)[:limit]
images = imageData.reshape(imageData.shape[:-1]+(28,28))
images_unattacked = [image.reshape(28, 28) for image in np.load(os.path.join(npys, examples, 'e0', 'advdata.npy')).astype(np.float64)[:limit]]

# Generates an unattacked image
def generateUnattackedImage(idx):
    fig = plt.figure()
    plt.imshow(images_unattacked[idx], cmap="gray")
    return fig

# get the adversarial data for a specific epsilon, caching it after the first time
def cached_get_data():
    cache = {}
    def __get_data(npys,eps):
        if eps not in cache:
            cache[eps] = np.load(os.path.join(npys,eps,'advdata.npy')).astype(np.float64)[:limit]
        return cache[eps]

    return __get_data
get_data = cached_get_data()

def getTrueLabel(idx):
    return exlabels[idx]

# cache the data for the nearest points as the same data will always be called at least twice if histogram is enabled
def cached_find_nearest():
    cache = {}
    def __findNearest(exdata,exoutput,advdata,idx,epsilon):
        if (idx,epsilon) not in cache:
            example = exdata[idx]
            label = np.argmax(exoutput[idx])

            norms = np.linalg.norm(advdata - example, axis=1)

            top = np.argpartition(norms, k-1)
            # cache norms of all data, the nearest k points, and the predicted label
            cache[(idx,epsilon)] = (norms, top[1:k], label)
        return cache[(idx,epsilon)]
    return __findNearest
findNearest = cached_find_nearest()

# create labels for histograms and sets labels to be visible only for the bottom one
def labelAxes(axs, plt):
    count = 0
    for ax in axs:
        ax.set_title(labels[count], fontstyle='italic', x = 0.8, y = 0.0)
        ax.get_xaxis().set_visible(False)
        count += 1
    axs[count-1].get_xaxis().set_visible(True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Generates an unlabeled image
def blitGenerateUnlabeledImage():
    def genUImg(idx):
        _,_,prediction = findNearest(exdata,exoutput,genUImg.advdata,idx,displayEpsilon)

        # set area where new text will go to blank
        genUImg.fig.canvas.restore_region(genUImg.titleBackground)

        # set image based on new data, and set prediction
        genUImg.img.set_data(images[idx])
        genUImg.title.set_text(f"Model prediction: {prediction}")

        # redraw image and title with altered data
        genUImg.ax.draw_artist(genUImg.img)
        genUImg.ax.draw_artist(genUImg.title)

        # update only the changed areas and flush updates
        genUImg.fig.canvas.blit(genUImg.ax.bbox)
        genUImg.fig.canvas.blit(genUImg.title.get_window_extent())
        genUImg.fig.canvas.flush_events()

        return genUImg.fig

    genUImg.advdata = get_data(npys,displayEpsilon)
    _,_,prediction = findNearest(exdata,exoutput,genUImg.advdata,0,displayEpsilon)

    # generate placeholder image and store figure, image, & bounding box of figure to load later
    genUImg.fig = plt.figure(tight_layout=True)
    genUImg.fig.set_size_inches(6/scaler, 4/scaler)
    genUImg.ax = genUImg.fig.add_subplot(1,1,1)
    genUImg.ax.set_xticks([])
    genUImg.ax.set_yticks([])

    # create blank title that is larger than area where new text will go, to get blank backgorund
    genUImg.title = genUImg.ax.set_title("                                ")

    genUImg.img = genUImg.ax.imshow(images[0], cmap="gray", interpolation="None")
    genUImg.fig.canvas.draw()
    # copy title background
    genUImg.titleBackground = genUImg.fig.canvas.copy_from_bbox(genUImg.title.get_window_extent())
    return genUImg
generateUnlabeledImage = blitGenerateUnlabeledImage()

# def generateUnlabeledImage(idx):
#     fig = plt.figure()
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(images[idx], cmap="gray")
#     return fig

# def generateTSNEPlots(idx):
#     origdata = get_data(npys,'e0')
#     advdata = get_data(npys,displayEpsilon)
#
#     norms,idxs,prediction = findNearest(exdata,exoutput,advdata,idx,displayEpsilon)
#
#     X_2d = []
#     if os.path.exists("./embedding.npy"):
#         X_2d = np.load('./embedding.npy').astype(np.float64)
#     else:
#         tsne = TSNE(n_components=2, random_state=4, perplexity=100)
#         X_2d = tsne.fit_transform(origdata)
#         np.save('./embedding.npy', X_2d, allow_pickle=False)
#
#     colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'
#
#     fig, (ax1,ax2) = plt.subplots(1,2,constrained_layout=True)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#
#     #plot embedding for class coloration
#     for c, label in zip(colors, labels):
#         ax1.scatter(X_2d[(testlabels[...] == label), 0], X_2d[(testlabels[...] == label), 1], c=c, label=label, s=3)
#
#     ax1.set_title("Test Data")
#
#     #plot embedding for norm coloration
#     scatterPlot = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=norms[...], s=3, cmap='viridis')
#
#     fig.colorbar(scatterPlot,label="norm")
#
#     #plot 10 nearest points
#     cb = ax2.scatter(X_2d[idxs,0],X_2d[idxs,1], c='red', s=3, zorder=2)
#     cb.set_clim(5,15)
#     # norms,idxs,prediction = findNearest(exdata,exoutput,advdata,idx+1,displayEpsilon)
#     # scatterPlot.set_array(norms)
#     # cb.set_offsets([[X_2d[i,0],X_2d[i,1]]for i in idxs])
#
#     # fig.colorbar(cb,label="norm")
#
#     ax2.set_title(f"Model Prediction: {prediction}\nAverage Distance: {round(float(sum(norms))/len(norms),2)}")
#
#     # cb.set_clim(min(norms),max(norms))
#     ax1.legend()
#     return fig

def blitgenerateTSNEPlots():
    def getTSNE(idx):
        # get closest points & norms to all points from the example at idx
        norms,idxs,prediction = findNearest(exdata,exoutput,getTSNE.advdata,idx,displayEpsilon)

        # restore backgrounds, clearing foregound and allowing redrawing of artists
        getTSNE.fig.canvas.restore_region(getTSNE.background)

        # change array for scatterplot so it'll recolor, changing offsets of cb so the closest 10 points will be in their new positions, and update title based on new model prediction
        getTSNE.scatterPlot.set_array(norms)
        getTSNE.cb.set_offsets([ [getTSNE.X_2d[i,0], getTSNE.X_2d[i,1]] for i in idxs])

        # redraw artists
        getTSNE.ax2.draw_artist(getTSNE.scatterPlot)
        getTSNE.ax2.draw_artist(getTSNE.cb)

        # update only the changed area and flush updates
        getTSNE.fig.canvas.blit(getTSNE.ax2.bbox)
        getTSNE.fig.canvas.flush_events()

        return getTSNE.fig
    # create figure/subplots, set widths so color bar will be much smaller than scatterplots, and turn off axes ticks
    getTSNE.fig, (getTSNE.ax1, getTSNE.ax2, getTSNE.ax3) = plt.subplots(1,3, tight_layout=True, gridspec_kw={'width_ratios': [10, 10, 1]},num="TSNE")
    getTSNE.fig.set_size_inches(6/scaler, 4/scaler)
    getTSNE.ax1.set_xticks([])
    getTSNE.ax1.set_yticks([])
    getTSNE.ax2.set_xticks([])
    getTSNE.ax2.set_yticks([])

    # load adversarial data for epsilon and get closest points to idx 0 for initial plot creation
    getTSNE.advdata = get_data(npys,displayEpsilon)
    norms,idxs,prediction = findNearest(exdata,exoutput,getTSNE.advdata,0,displayEpsilon)

    # generate tsne embedding based on original set of data
    X_2d = []
    if os.path.exists("./embedding.npy"):
        X_2d = np.load('./embedding.npy').astype(np.float64)
    else:
        tsne = TSNE(n_components=2, random_state=4, perplexity=100)
        origdata = get_data(npys,'e0')
        X_2d = tsne.fit_transform(origdata)
        np.save('./embedding.npy', X_2d, allow_pickle=False)
    getTSNE.X_2d = X_2d

    # create scatter of all points colored & labaled by class. this one never needs to be updated, it's static
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'
    for c, label in zip(colors, labels):
        getTSNE.ax1.scatter(X_2d[(testlabels[:] == label), 0], X_2d[(testlabels[:] == label), 1], c=c, label=label, s=3)

    # set static title for both hists and legends for first hist
    getTSNE.ax1.set_title("Class Labels")
    getTSNE.ax2.set_title("Distance heatmap")
    lgnd = getTSNE.ax1.legend(loc='lower left', framealpha=1, handletextpad=0.2, scatteryoffsets=[0.5], labelspacing=0.25, borderpad=0.3, borderaxespad=0.25, handlelength=1.1, bbox_to_anchor=(-0.1, -0.0125), edgecolor='black')
    for label in lgnd.legendHandles:
        label._sizes = [30]

    colorLim = (4,13)

    # manually create colorbar before second scatterplot has been made
    getTSNE.fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=colorLim[0],vmax=colorLim[1]),cmap='viridis'),cax=getTSNE.ax3,label="norm")

    # store bounding boxes of scatterplot background and draw figure
    getTSNE.background = getTSNE.fig.canvas.copy_from_bbox(getTSNE.ax2.bbox)
    getTSNE.fig.canvas.draw()

    # create scatter plot of all data colored by example's distance from original data & closest 10 points
    getTSNE.scatterPlot = getTSNE.ax2.scatter(X_2d[:,0], X_2d[:,1], c=norms[:], s=1, cmap='viridis', zorder=1)
    getTSNE.cb = getTSNE.ax2.scatter(X_2d[idxs,0],X_2d[idxs,1], c='red', s=5, zorder=2)
    getTSNE.scatterPlot.set_clim(colorLim[0],colorLim[1])

    return getTSNE
generateTSNEPlots = blitgenerateTSNEPlots()

def roundSigFigs(num, sigFigs):
    # convert num to string and cut end based on # of significant figures
    # if num is 0, only print single digit. Else cut based on # of significant figures, +2 for the decimal & first full digit, and if it's greater than 1 log10 of num
    return str(num)[:(int(math.log(num,10))*(num>1) + 2 + sigFigs if num else 1)]

maxEpsilon = config["General"]["maxEpsilon"]
epsilonStepSize = config["General"]["epsilonStepSize"]
# finds # of significant figures after the decimal place of the step size
sigFigs = len(repr(float(epsilonStepSize)).split('.')[1].rstrip('0'))
epsilonList = generateEpsilonList(epsilonStepSize,maxEpsilon)
def generateHistograms(idx, plotID, height = None):
    r=(5,16)
    b=150

    maxHeight = 0
    subplot_create = time.time()
    fig, axs = plt.subplots(10)
    fig.set_size_inches(6/scaler, 4/scaler)
    print("Create subplots:",time.time()-subplot_create)

    generateHist = time.time()
    cmap = plt.get_cmap("tab10")
    if plotID > maxEpsilon:
        for epsilon in epsilonList:
            advdata = get_data(npys,f'e{roundSigFigs(epsilon,sigFigs)}')
            norms,_,_ = findNearest(exdata,exoutput,advdata,idx,epsilon)

            colorIdx = epsilonList.index(epsilon)

            for i in range(10):
                arr = norms[(testlabels[...] == labels[i])]
                weights = np.ones_like(arr)/len(arr)
                if i:
                    y, _, _ = axs[i].hist(arr, weights=weights, color=cmap(colorIdx), alpha=0.5, bins=b,range=r,density=False, histtype="step")
                else:
                    y, _, _ = axs[i].hist(arr, weights=weights, color=cmap(colorIdx), alpha=0.5, bins=b,range=r,density=False, histtype="step", label=f"Attack Strength {epsilon}")
                maxHeight = max(maxHeight,y.max())
        fig.legend(loc='upper left')
        fig.suptitle("All Attack Strengths")
    else:
        advdata = get_data(npys,f'e{roundSigFigs(plotID,sigFigs)}')
        norms,_,_ = findNearest(exdata,exoutput,advdata,idx,plotID)

        colorIdx = epsilonList.index(plotID)
        for i in range(10):
            arr = norms[(testlabels[...] == labels[i])]
            weights = np.ones_like(arr)/len(arr)
            y, _, _ = axs[i].hist(arr, weights=weights, color=cmap(colorIdx), alpha=0.5, bins=b,range=r,density=False, label=f"Attack Strength {plotID}", histtype="step")
            maxHeight = max(maxHeight,y.max())
        fig.suptitle(f"Attack Strength {plotID}")

    for ax in axs:
        ax.set_ylim([0, height if height else maxHeight])

    labelAxes(axs, fig)
    print("Histogram generation:",time.time()-generateHist)
    if height:
        return fig
    return (fig, maxHeight)

def generateBoxPlot(idx):
    fig, axs = plt.subplots()
    fig.set_size_inches(6/scaler, 4/scaler)
    norm_list = []

    for epsilon in epsilonList:
        advdata = get_data(npys, f'e{roundSigFigs(epsilon,sigFigs)}')
        norms,_,prediction = findNearest(exdata, exoutput, advdata, idx, epsilon)
        norm_list.append(norms)

    plt.suptitle(f"Model Prediction: {prediction}")
    axs.boxplot(norm_list, patch_artist=True, notch='True', vert=1, labels=[f"Epsilon {epsilon}" for epsilon in epsilonList], showmeans=True)
    axs.axis([None,None,3,15.5])
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
    pt_state_dict = torch.load('./model/' + attack_type + '/chained_ae_' + dist_metric + '_{}_to_{}_d={}_epc{}.pth'.format(6, 0, d, 500))
    ae = load_part_of_model(ae, pt_state_dict, numblocks)
    ae.to(device)
    ae.eval()

    return ae

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

        # calculate current batch number and where in that batch based on current index compared to starting index and preset batch size
        localIdx = (idx - __trajectoryCostReg.startIdx) % batch_size
        batchNum = (idx - __trajectoryCostReg.startIdx) // batch_size

        # if new index requires new batch, cut out new batch from loaded data, generate cost regressions, and round cost regressions
        if not localIdx:
             temp = images[__trajectoryCostReg.startIdx + batchNum*batch_size:__trajectoryCostReg.startIdx + batchNum*batch_size + batch_size]
             __trajectoryCostReg.batchData = torch.unsqueeze(torch.from_numpy(temp),1).to(torch.float)
             __trajectoryCostReg.pc = cost_reg(__trajectoryCostReg.batchData).detach().cpu().numpy()
             __trajectoryCostReg.rounded_pc = round_cost(__trajectoryCostReg.pc)

        exp = torch.unsqueeze(__trajectoryCostReg.batchData[localIdx], 0)
        # get cost of current index
        cost = __trajectoryCostReg.rounded_pc[localIdx]

        # create models and perform Reconstruction of current example
        reg_epsilons = range(int(cost)+1)
        reg_ae = make_models(dist_metric, reg_epsilons, attack_type, d)
        reg_recons = reg_ae(exp,True)

        fig = plt.figure()
        fig.set_size_inches(6/scaler, 4/scaler)
        fig.suptitle('Predicted Attack Strength: ({:.2f})'.format(__trajectoryCostReg.pc[localIdx].item()))

        # embed reconstructed images and label them
        num_bins = len(reg_recons)+1
        for j in range(num_bins):
            ax = fig.add_subplot(1,num_bins,j+1,anchor='N')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title(f'Attack\nStrength {reg_epsilons[j]}', fontsize=10)
            # outline original image in red
            if j == len(reg_recons):
                ax.imshow(exp[0][0], cmap="gray")
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            else:
                ax.imshow(reg_recons[num_bins-2-j][0][0].detach().cpu(), cmap="gray")

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
