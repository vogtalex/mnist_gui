import os
import torch
from torch.utils.data import DataLoader
from functions import shuffle_together, generateEpsilonList, PGD_attack
from mnist_cost import MNISTCost
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
import json
import sys
import shutil
import math

with open('config.json') as f:
   config = json.load(f)

numIters = 20
batchSize = 1000
use_cuda = True

# epsilons for subset to be generated with
subset_eps = [3,6]

### import model from variable directory
# split file into name and path
head,tail = os.path.split(config['Model']['modelDir'])
# temporarily add directory where python file is present to path
sys.path.append(head)
# format import with specific module name and execute the import
exec(f"from {tail[:-3]} import *")

# get file output directory from config
outputDir = config['Model']['outputDir']
# Use a pretrained model
pretrained_model = config["Model"]["weightDir"]

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batchSize, shuffle=True)

# use cuda if available and var is set
device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = MadryNet()
model.load_state_dict(torch.load(pretrained_model,map_location=device))
model.to(device)

torch.backends.cudnn.enabled = False # I don't know what this does

loss_fn = nn.functional.cross_entropy

# generates the "known" data
def gen_adv_features_test(eps):
    model.eval()
    cnt = 0

    labels = []
    out_data = np.array([])
    out_output = np.array([])

    for data, target in test_loader:
        cnt += 1

        # break loop after going through 90% of the total examples
        if cnt > (len(test_loader.dataset)/batchSize)*0.9: break;
        print("processing: %d/%d" % (cnt*batchSize, math.floor(len(test_loader.dataset)*0.9)))
        data,target = data.to(device), target.to(device)

        # if not epsilon 0, generate attacked image
        if eps:
            attackedImg = PGD_attack(model=model, device=device, loss=loss_fn, x=data, y=target, epsilon=eps, niter=numIters, stepsize=2.5*eps/numIters, lpnorm=2, randinit=True, debug=False)
        else:
            attackedImg = data.detach()

        # generate model output from example and append to other examples
        output = model(attackedImg)
        output_np = output.data.cpu().numpy()
        out_output = np.vstack([out_output,output_np]) if out_output.size else output_np

        labels = np.append(labels,target.cpu().numpy())

        # reshape attacked image tensor and add to output
        attackedImg = torch.flatten(attackedImg,2,3)
        attackedImg = torch.flatten(attackedImg,0,1)
        out_data = np.vstack([out_data,attackedImg.cpu().numpy()]) if out_data.size else attackedImg.cpu().numpy()

    print("Finished test images for epsilon")
    labelPath = os.path.join(outputDir,"testlabels.npy")
    # only save labels once, since all epsilon sets share same true labels
    if not os.path.isfile(labelPath):
        np.save(labelPath, labels, allow_pickle=False)
    os.mkdir(os.path.join(outputDir,f"e{eps}"))
    np.save(os.path.join(outputDir,f"e{eps}","advoutput.npy"), out_output, allow_pickle=False)
    np.save(os.path.join(outputDir,f"e{eps}","advdata.npy"), out_data, allow_pickle=False)

# generate the "unkown" data
def gen_adv_features_examples(numSubsets,subsetSize):
    model.eval()
    cnt = 0

    currentSubset = 0

    # load cost estimator
    cost_reg = MNISTCost()
    cost_reg.load_state_dict(torch.load('./model/MNIST-Cost_est_l2.pth',map_location=torch.device(device)))
    cost_reg.to(device)
    cost_reg.train()

    for data, target in test_loader:
        cnt += 1
        # skip until last 10% of dataset (so it's separate dataset from "test" set)
        if cnt <= (len(test_loader.dataset)/batchSize)*0.9: continue;
        # finish if generated number of requested subsets
        if currentSubset >= numSubsets: break;

        data,target = data.to(device), target.to(device)

        idx=0
        # if generating new subset would index out of batch, or if all subsets are generated, exit loop
        while((idx+1)*subsetSize < batchSize*0.1 and currentSubset<numSubsets):
            labels = []
            out_data = np.array([])
            out_output = np.array([])
            out_pc = np.array([])
            data_subset_whole = np.array([])

            for eps in subset_eps:
                # create subsets of the data, splitting the subset based on how many epsilons are going to be generated
                data_subset = data[math.floor(idx*subsetSize):math.floor((idx+1/len(subset_eps))*subsetSize)]
                target_subset = target[math.floor(idx*subsetSize):math.floor((idx+1/len(subset_eps))*subsetSize)]

                # generate attacked examples
                if eps:
                    attackedImg = PGD_attack(model=model, device=device, loss=loss_fn, x=data_subset, y=target_subset, epsilon=eps, niter=numIters, stepsize=2.5*eps/numIters, lpnorm=2, randinit=True, debug=False)
                else:
                    attackedImg = data_subset.detach()

                # append model output to outputs for subset
                output = model(attackedImg)
                out_output = np.vstack([out_output,output.data.cpu().numpy()]) if out_output.size else output.data.cpu().numpy()

                labels = np.append(labels,target_subset.cpu().numpy())

                # append attacked image to output for subset
                attackedImg = torch.flatten(attackedImg,2,3)
                attackedImg = torch.flatten(attackedImg,0,1)
                out_data = np.vstack([out_data,attackedImg.cpu().numpy()]) if out_data.size else attackedImg.cpu().numpy()

                # append unattacked image to output for subset
                data_subset = torch.flatten(data_subset,2,3)
                data_subset = torch.flatten(data_subset,0,1)
                data_subset_whole = np.vstack([data_subset_whole,data_subset.cpu().numpy()]) if data_subset_whole.size else data_subset.cpu().numpy()

                # append cost estimates to output for subset
                reshapedData = attackedImg.reshape(attackedImg.shape[:-1]+(28,28))
                batchData = torch.unsqueeze(reshapedData,1).to(torch.float)
                pc = cost_reg(batchData).detach().cpu().numpy().flatten()
                out_pc = np.append(out_pc,pc) if out_pc.size else pc

                # increment index by fraction of subset, based on how many epsilons the subset is being split into
                idx += 1/len(subset_eps)

            # shuffle all output arrays with same randomness (since they're all same length, results in same shuffle)
            shuffle_together(labels,out_data,out_output,data_subset_whole,out_pc)

            print(f"Generated subset {currentSubset}")
            # make examples folder if it doesn't already exist
            folderPath = os.path.join(outputDir,'examples')
            if not os.path.isdir(folderPath):
                os.mkdir(folderPath)
            # save all generated arrays for subset
            os.mkdir(os.path.join(outputDir,'examples',f'subset{currentSubset}'))
            np.save(os.path.join(outputDir,'examples',f'subset{currentSubset}',"testlabels.npy"), labels, allow_pickle=False)
            np.save(os.path.join(outputDir,'examples',f'subset{currentSubset}',"advoutput.npy"), out_output, allow_pickle=False)
            np.save(os.path.join(outputDir,'examples',f'subset{currentSubset}',"advdata.npy"), out_data, allow_pickle=False)
            np.save(os.path.join(outputDir,'examples',f'subset{currentSubset}',"data.npy"), data_subset_whole, allow_pickle=False)
            np.save(os.path.join(outputDir,'examples',f'subset{currentSubset}',"pc.npy"), out_pc, allow_pickle=False)

            currentSubset += 1

# clear output directory
try:
    shutil.rmtree(outputDir)
except OSError as e:
    print("Error: %s : %s" % (outputDir, e.strerror))
os.mkdir(outputDir)

# generate requested number of subsets of requested size
gen_adv_features_examples(config['Model']['numSubsets'],config['Model']['subsetSize'])

# generate full epsilon sets for known data
for eps in generateEpsilonList(config["General"]["epsilonStepSize"], config["General"]["maxEpsilon"]):
    print(f"Generating epsilon {eps} data")
    gen_adv_features_test(eps)
