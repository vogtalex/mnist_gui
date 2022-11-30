# Load Python Libraries
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
from sklearn.manifold import TSNE
import json
import math
from functions import generateEpsilonList
import torch
import torch.nn as nn
import torch.optim
from collections import OrderedDict
from chained_AE import Autoencoder, Chained_AE
from mnist_cost import MNISTCost

with open('config.json') as f:
   config = json.load(f)

# get pre-generated data
try:
    npys = config['Model']['outputDir']
except:
    exit("Data path not valid")

# which subset to use for experiment run
displaySubset = f"subset{config['General']['displaySubset']}"
examples = 'examples'

# figure size downscale value
scaler = 1
# num of closest points for tsne
k=10

# constants for trajectory regression
use_cuda = False
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
dist_metric = 'l2'
attack_type = 'untargeted'
d=50
max_epsilon = 6
batch_size = 10 # MUST NOT BE LOWER THAN 10

#load data for example subset
exlabels = np.load(os.path.join(npys,examples,displaySubset,'testlabels.npy'),mmap_mode='r').astype(np.float64)
exoutput = np.load(os.path.join(npys,examples,displaySubset,'advoutput.npy'),mmap_mode='r').astype(np.float64)
exdata = np.load(os.path.join(npys,examples,displaySubset,'advdata.npy'),mmap_mode='r').astype(np.float64)
data = np.load(os.path.join(npys, examples,displaySubset, 'data.npy'),mmap_mode='r').astype(np.float64)
pc = np.load(os.path.join(npys, examples,displaySubset, 'pc.npy'),mmap_mode='r').astype(np.float64)

images = exdata.reshape(exdata.shape[:-1]+(28,28))
images_unattacked = data.reshape(data.shape[:-1]+(28,28))

testlabels = np.load(os.path.join(npys,'testlabels.npy'),mmap_mode='r').astype(np.float64)

# this is kinda a makeshift solution, do it better later
labels = list(map(int,set(testlabels)))

# Generates a plot w/ the unattacked image
def generateUnattackedImage(idx):
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(6/scaler, 4/scaler)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images_unattacked[idx], cmap="gray")
    return fig

# get the adversarial data for a specific epsilon, caching it after the first time
def cached_get_data():
    cache = {}
    def __get_data(npys,eps):
        if eps not in cache:
            cache[eps] = np.load(os.path.join(npys,eps,'advdata.npy'),mmap_mode='r').astype(np.float64)
        return cache[eps]

    return __get_data
get_data = cached_get_data()

def getTrueLabel(idx):
    return exlabels[idx]

# calculate l2 norm between attacked example and unattacked example
def getAttackStrength(idx):
    return np.linalg.norm(data[idx]-exdata[idx])

# find the k nearest points to example at idx, calulcate norms, & get norms
def cached_find_nearest():
    cache = {}
    def __findNearest(exdata,exoutput,advdata,idx,epsilon):
        if (idx,epsilon) not in cache:
            example = exdata[idx]
            label = np.argmax(exoutput[idx])

            norms = np.linalg.norm(advdata - example, axis=1)

            top = np.argpartition(norms, k-1)
            # cache norms of all data, the nearest k points, and the predicted label
            cache[(idx,epsilon)] = (norms, top[0:k], label)
            return (norms, top[0:k], label)
        return cache[(idx,epsilon)]
    return __findNearest
findNearest = cached_find_nearest()

# create labels for histograms and sets labels to be visible only for the bottom hist
def labelAxes(axs, plt):
    for i in range(len(axs)):
        axs[i].set_title(labels[i], fontstyle='italic', x = 0.7, y = 0.0)
        axs[i].get_xaxis().set_visible(False)
    axs[len(axs)-1].get_xaxis().set_visible(True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Generates an unlabeled image
def blitGenerateUnlabeledImage():
    def genUImg(idx):
        _,_,prediction = findNearest(exdata,exoutput,genUImg.advdata,idx,-1)

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

    genUImg.advdata = get_data(npys,'e0')

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

def blitgenerateTSNEPlots():
    def getTSNE(idx):
        # get closest points & norms to all points from the example at idx
        norms,idxs,prediction = findNearest(exdata,exoutput,getTSNE.data,idx,-1)

        # restore backgrounds, clearing foregound and allowing redrawing of artists
        getTSNE.fig.canvas.restore_region(getTSNE.background)

        # change array for scatterplot so it'll recolor and change offsets of cb so the closest 10 points will be in their new positions
        getTSNE.scatterPlot.set_array(norms)
        getTSNE.cb.set_offsets(getTSNE.X_2d[idxs])

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

    # load unattacked data for epsilon and get closest points to idx 0 for initial plot creation
    getTSNE.data = get_data(npys,'e0')
    norms,idxs,_ = findNearest(exdata,exoutput,getTSNE.data,0,-1)

    # generate tsne embedding based on original set of data
    X_2d = []
    if os.path.exists("./embedding.npy"):
        X_2d = np.load('./embedding.npy').astype(np.float64)
    else:
        tsne = TSNE(n_components=2, random_state=4, perplexity=100)
        X_2d = tsne.fit_transform(getTSNE.data)
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

    colorLim = (4,14)

    # manually create colorbar before second scatterplot has been made
    getTSNE.fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=colorLim[0],vmax=colorLim[1]),cmap='viridis'),cax=getTSNE.ax3)

    # draw figure and save background of scatterplot
    getTSNE.fig.canvas.draw()
    getTSNE.background = getTSNE.fig.canvas.copy_from_bbox(getTSNE.ax2.bbox)

    # create scatter plot of all data colored by example's distance from original data & closest 10 points
    getTSNE.scatterPlot = getTSNE.ax2.scatter(X_2d[:,0], X_2d[:,1], c=norms[:], s=1, cmap='viridis', zorder=1)
    getTSNE.cb = getTSNE.ax2.scatter(X_2d[idxs,0],X_2d[idxs,1], c='red', s=5, zorder=2)
    getTSNE.scatterPlot.set_clim(*colorLim)

    return getTSNE
generateTSNEPlots = blitgenerateTSNEPlots()

def roundSigFigs(num, sigFigs):
    # convert num to string and cut end based on # of significant figures
    # if num is 0, only print single digit. Else cut based on # of significant figures, +2 for the decimal & first full digit, and if it's greater than 1 +log10 of num
    return str(num)[:(int(math.log(num,10))*(num>1) + 2 + sigFigs if num else 1)]

maxEpsilon = config["General"]["maxEpsilon"]
epsilonStepSize = config["General"]["epsilonStepSize"]
# finds # of significant figures after the decimal place of the step size
sigFigs = len(repr(float(epsilonStepSize)).split('.')[1].rstrip('0'))

epsilonList = generateEpsilonList(epsilonStepSize,maxEpsilon)
def blitGenerateHistograms():
    r=(4,14)
    b=150

    def genHist(idx, eps, height = None):
        maxHeight = 0
        # all epsilon histogram
        if eps > maxEpsilon:
            # get figure from cache
            (fig,axs,histObjs) = genHist.figCache[len(epsilonList)]
            for epsilon in epsilonList:
                # load adversarial data & calculate norms to example
                advdata = get_data(npys,f'e{roundSigFigs(epsilon,sigFigs)}')
                norms,_,_ = findNearest(exdata,exoutput,advdata,idx,epsilon)
                figIdx = epsilonList.index(epsilon)
                for i in range(len(labels)):
                    # if histogram hasn't been cached yet, generate hist for it & cache it
                    if (idx,epsilon,i) not in genHist.histCache:
                        arr = norms[(testlabels[...] == labels[i])]
                        weights = np.ones_like(arr)/len(arr)

                        n, _ = np.histogram(arr,bins=b,range=r,weights=weights)
                        genHist.histCache[(idx,epsilon,i)] = n
                    # otherwise pull generated data out of cache
                    else:
                        n = genHist.histCache[(idx,epsilon,i)]

                    # set heights of hist to new histogram data
                    maxHeight = max(maxHeight,n.max())
                    histObjs[i][figIdx][0].set_ydata(n)

        # individual epsilon histogram
        else:
            # get figure from cache
            figIdx = epsilonList.index(eps)
            (fig,axs,histObjs) = genHist.figCache[figIdx]

            # load adversarial data & calculate norms to example
            advdata = get_data(npys,f'e{roundSigFigs(eps,sigFigs)}')
            norms,_,_ = findNearest(exdata,exoutput,advdata,idx,eps)

            for i in range(len(labels)):
                # if histogram hasn't been cached yet, generate hist for it & cache it
                if (idx,eps,i) not in genHist.histCache:
                    arr = norms[(testlabels[...] == labels[i])]
                    weights = np.ones_like(arr)/len(arr)

                    n, _ = np.histogram(arr,bins=b,range=r,weights=weights)
                    genHist.histCache[(idx,eps,i)] = n
                # otherwise pull generated data out of cache
                else:
                    n = genHist.histCache[(idx,eps,i)]

                # set heights of hist to new histogram data
                maxHeight = max(maxHeight,n.max())
                histObjs[i][0].set_ydata(n)

        # set max height of all axis, either to calculated max height or height argument if provided
        for ax in axs:
            ax.set_ylim([0, height if height else maxHeight])

        # redraw figure w/ changes and flush changes
        fig.canvas.draw()
        fig.canvas.flush_events()

        # if no height argument was given return calculated maxheight
        if height:
            return fig
        return (fig, maxHeight)

    cmap = plt.get_cmap("tab10")

    genHist.histCache = {}
    genHist.figCache = [None] * (len(epsilonList)+1)

    # populate figure cache with figures for each single epsilon plot
    for epsilon in epsilonList:
        advdata = get_data(npys,f'e{roundSigFigs(epsilon,sigFigs)}')
        norms,_,_ = findNearest(exdata,exoutput,advdata,0,epsilon)

        # get index of epsilon in list for line color
        colorIdx = epsilonList.index(epsilon)

        fig, axs = plt.subplots(10)
        fig.set_size_inches(6/scaler, 4/scaler)
        histObjs = []

        for i in range(len(labels)):
            arr = norms[(testlabels[...] == labels[i])]
            weights = np.ones_like(arr)/len(arr)

            n, edges = np.histogram(arr,bins=b,range=r,weights=weights)
            # cache height values for histogram
            genHist.histCache[(0,epsilon,i)] = n

            # create matplotlib hist from generated numpy vals
            newHist = axs[i].step(edges[:-1],n,where='post',color=cmap(colorIdx),alpha=0.5, label=f"Attack Strength {epsilon}")
            axs[i].set_xlim(r)
            histObjs.append(newHist)

        fig.suptitle(f"Attack Strength {epsilon}")
        labelAxes(axs, fig)
        # cache figure and reference to histograms for epsilon index
        genHist.figCache[colorIdx] = (fig,axs,histObjs)

    # populate and cache all epsilon plot
    fig, axs = plt.subplots(10)
    fig.set_size_inches(6/scaler, 4/scaler)
    histObjs = [[] for _ in range(len(labels))]
    for epsilon in epsilonList:
        colorIdx = epsilonList.index(epsilon)

        # since all data we need was already generated by single hist creation, just pull it out of cache and plot it together
        for i in range(len(labels)):
            n = genHist.histCache[(0,epsilon,i)]
            if i:
                newHist = axs[i].step(edges[:-1],n,where='post',color=cmap(colorIdx),alpha=0.5)
            else:
                newHist = axs[i].step(edges[:-1],n,where='post',color=cmap(colorIdx),alpha=0.5, label=f"Attack Strength {epsilon}")
            axs[i].set_xlim(r)
            histObjs[i].append(newHist)

    fig.legend(loc='upper right')
    fig.suptitle("All Attack Strengths")
    labelAxes(axs, fig)
    # cache figure and references to arrays of hists for greater than max epsilon index (which is used for all eps hist)
    genHist.figCache[len(epsilonList)] = (fig,axs,histObjs)

    return genHist
generateHistograms = blitGenerateHistograms()

# generate box plot visualization (currently unused)
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
    pt_state_dict = torch.load('./model/' + attack_type + '/chained_ae_' + dist_metric + '_{}_to_{}_d={}_epc{}.pth'.format(6, 0, d, 500),map_location=torch.device(device))
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
        # calculate current batch number and where in that batch based on current index compared to starting index and preset batch size
        localIdx = (idx - __trajectoryCostReg.startIdx) % batch_size
        batchNum = (idx - __trajectoryCostReg.startIdx) // batch_size

        # if new index requires new batch, cut out new batch from loaded data, generate cost regressions, and round cost regressions
        if not localIdx:
             temp = images[__trajectoryCostReg.startIdx + batchNum*batch_size:__trajectoryCostReg.startIdx + (batchNum+1)*batch_size]
             __trajectoryCostReg.batchData = torch.unsqueeze(torch.from_numpy(temp),1).to(torch.float)
             __trajectoryCostReg.pc = pc[__trajectoryCostReg.startIdx + batchNum*batch_size:__trajectoryCostReg.startIdx + (batchNum+1)*batch_size]
             __trajectoryCostReg.rounded_pc = round_cost(__trajectoryCostReg.pc)

        exp = torch.unsqueeze(__trajectoryCostReg.batchData[localIdx], 0)
        cost = __trajectoryCostReg.rounded_pc[localIdx]

        # create models and perform Reconstruction for current example
        reg_epsilons = range(int(cost)+1)
        reg_ae = make_models(dist_metric, reg_epsilons, attack_type, d)
        reg_recons = reg_ae(exp,True)

        __trajectoryCostReg.fig.clf()
        __trajectoryCostReg.fig.suptitle('Predicted Attack Strength: ({:.2f})'.format(__trajectoryCostReg.pc[localIdx].item()))

        # embed reconstructed images and label them
        num_bins = len(reg_recons)+1
        for j in range(num_bins):
            ax = __trajectoryCostReg.fig.add_subplot(1,num_bins,j+1,anchor='N')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Attack\nStrength {reg_epsilons[j]}', fontsize=10)
            # embed images and outline original image in red
            if j == len(reg_recons):
                ax.imshow(exp[0][0], cmap="gray")
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            else:
                ax.imshow(reg_recons[num_bins-2-j][0][0].detach().cpu(), cmap="gray")

        __trajectoryCostReg.fig.canvas.draw()
        __trajectoryCostReg.fig.canvas.flush_events()
        return __trajectoryCostReg.fig

    __trajectoryCostReg.fig = plt.figure(tight_layout=True)
    __trajectoryCostReg.fig.set_size_inches(6/scaler, 4/scaler)

    # important to set start index for relative indexing into batch
    __trajectoryCostReg.startIdx = idx
    __trajectoryCostReg.batchData = None
    __trajectoryCostReg.pc = 0
    __trajectoryCostReg.rounded_pc = 0
    global trajectoryCostRegFunc
    trajectoryCostRegFunc = __trajectoryCostReg

def trajectoryCostReg(idx):
    return trajectoryCostRegFunc(idx)
