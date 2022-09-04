import os
import torch
from torch.utils.data import DataLoader
from functions import *
import numpy as np

# check if swap works
import torch.nn as nn
# from torch.nn import CrossEntropyLoss

# check if swap works
import torch.optim as optim
# from torch.optim import SGD

from torchvision import datasets, transforms
import json
import sys
import shutil

with open('config.json') as f:
   config = json.load(f)

numIters = 20
batchSize = 1000
use_cuda = True

### import model from variable directory and import it
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
    batch_size=batchSize, shuffle=False)

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
#         transforms.ToTensor(),
#     ])),
#     batch_size=batchSize, shuffle=True)

# use cuda if available and var is set
device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = MadryNet()
model.load_state_dict(torch.load(pretrained_model,map_location=device))
model.to(device)

# Creates a random seed
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Creates loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def gen_adv_features_test(eps):
    model.eval()
    cnt = 0

    labels = []
    out_data = np.array([])
    out_output = np.array([])

    testLoaderLen = len(test_loader.dataset)

    for data, target in test_loader:
        cnt += 1
        if cnt > (testLoaderLen/batchSize)*0.9: break;
        print("processing: %d/%d" % (cnt*batchSize, testLoaderLen))
        data,target = data.to(device), target.to(device)

        # delta = pgd_l2(model,data,target,eps,2.5*eps/255./numIters,numIters)
        if eps:
            attackedImg = PGD_attack(model=model, device=device, loss=F.cross_entropy, x=data, y=target, epsilon=eps, niter=numIters, stepsize=2.5*eps/numIters, lpnorm=2, randinit=False, debug=False)
        else:
            attackedImg = data.detach()

        output = model(attackedImg)
        output_np = output.data.cpu().numpy()
        out_output = np.vstack([out_output,output_np]) if out_output.size else output_np

        labels = np.append(labels,target.cpu().numpy())

        data = torch.flatten(data,2,3)
        data = torch.flatten(data,0,1)
        out_data = np.vstack([out_data,data.cpu().numpy()]) if out_data.size else data.cpu().numpy()

    print("Finished test images for epsilon")
    labelPath = os.path.join(outputDir,"testlabels.npy")
    if not os.path.isfile(labelPath):
        np.save(labelPath, labels, allow_pickle=False)
    os.mkdir(os.path.join(outputDir,f"e{eps}"))
    np.save(os.path.join(outputDir,f"e{eps}","advoutput.npy"), out_output, allow_pickle=False)
    np.save(os.path.join(outputDir,f"e{eps}","advdata.npy"), out_data, allow_pickle=False)

def gen_adv_features_examples(eps):
    model.eval()
    cnt = 0

    labels = []
    out_data = np.array([])
    out_output = np.array([])

    testLoaderLen = len(test_loader.dataset)

    for data, target in test_loader:
        cnt += 1
        if cnt <= (testLoaderLen/batchSize)*0.9: continue;
        print("processing: %d/%d" % (cnt*batchSize, testLoaderLen))
        data,target = data.to(device), target.to(device)

        # delta = pgd_l2(model,data,target,eps,2.5*eps/255./numIters,numIters)
        if eps:
            attackedImg = PGD_attack(model=model, device=device, loss=F.cross_entropy, x=data, y=target, epsilon=eps, niter=numIters, stepsize=2.5*eps/numIters, lpnorm=2, randinit=False, debug=False)
        else:
            attackedImg = data.detach()

        output = model(attackedImg)
        output_np = output.data.cpu().numpy()
        out_output = np.vstack([out_output,output_np]) if out_output.size else output_np

        labels = np.append(labels,target.cpu().numpy())

        attackedImg = torch.flatten(attackedImg,2,3)
        attackedImg = torch.flatten(attackedImg,0,1)
        out_data = np.vstack([out_data,attackedImg.cpu().numpy()]) if out_data.size else attackedImg.cpu().numpy()

    print("Finished example images for epsilon")
    folderPath = os.path.join(outputDir,'examples')
    if not os.path.isdir(folderPath):
        os.mkdir(folderPath)
    labelPath = os.path.join(outputDir,'examples',"testlabels.npy")
    if not os.path.isfile(labelPath):
        np.save(labelPath, labels, allow_pickle=False)
    os.mkdir(os.path.join(outputDir,'examples',f"e{eps}"))
    np.save(os.path.join(outputDir,'examples',f"e{eps}","advoutput.npy"), out_output, allow_pickle=False)
    np.save(os.path.join(outputDir,'examples',f"e{eps}","advdata.npy"), out_data, allow_pickle=False)

# clear output directory
try:
    shutil.rmtree(outputDir)
except OSError as e:
    print("Error: %s : %s" % (outputDir, e.strerror))
os.mkdir(outputDir)

for eps in generateEpsilonList(config["General"]["epsilonStepSize"], config["General"]["maxEpsilon"]):
    print(f"Generating epsilon {eps} data")
    gen_adv_features_test(eps)
    gen_adv_features_examples(eps)
