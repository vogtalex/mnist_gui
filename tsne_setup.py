
from itertools import islice
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from models.net import *

from functions import *

import torch
import numpy as np
import os
#import matplotlib.pyplot as plt

import time
import tkinter as tk

import torch
import torch.onnx as onnx
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torchvision
from torch.autograd import Variable

import tkinter as tk
from tkinter import *

from PIL import Image, ImageTk

from functools import partial


from numpy import load


# mtpltlib bug:
matplotlib.use('TkAgg')


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=False)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=True)

# Starter code if using cuda
use_cuda = True
device = torch.device("cuda:0" if (
    use_cuda and torch.cuda.is_available()) else "cpu")

# Use a pretrained model
pretrained_model = "lenet_mnist_model.pth"

# Initialize the network
model = Net()
model.load_state_dict(torch.load(pretrained_model, map_location=device))

# Creates a random seed
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Creates loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# FGSM attack code


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def gen_adv_features_test():
    model.eval()
    cnt = 0

    labels = []

    out_data = []
    out_adv_data = []

    #out_target = []

    #out_output = []
    out_adv_output = []

    for data, target in test_loader:
        cnt += 1
        print("processing: %d/%d" % (cnt, len(test_loader.dataset)))
        if cnt > 9000:
            break

        #delta = fgsm(model,data,target,0.2)
        delta = pgd_linf(model, data, target, 0.05, 1e-2, 40)

        #output = model(data)
        adv_output = model(data + delta)

        #output_np = output.data.cpu().numpy()
        adv_output_np = adv_output.data.cpu().numpy()
        # print(np.argmax(output_np),np.argmax(adv_output_np))

        labels.append(target.numpy()[0])
        #target_np = output.max(1, keepdim=True)[1][0].numpy()

        adv_data = data+delta
        ######################

        # out_output.append(output_np)
        out_adv_output.append(adv_output_np)

        #out_target.append(target_np[:, np.newaxis])

        #data = torch.flatten(data,2,3)
        #data = torch.flatten(data,0)
        # out_data.append(data.numpy())

        adv_data = torch.flatten(adv_data, 2, 3)
        adv_data = torch.flatten(adv_data, 0)
        out_adv_data.append(adv_data.numpy())

    #output_array = np.concatenate(out_output, axis=0)
    adv_output_array = np.concatenate(out_adv_output, axis=0)
    #target_array = np.concatenate(out_target, axis=0)

    #np.save('./npys/output.npy', output_array, allow_pickle=False)
    np.save('./npys/e1/testlabels.npy', labels, allow_pickle=False)
    np.save('./npys/e1/advoutput.npy', adv_output_array, allow_pickle=False)
    #np.save('./npys/target.npy', target_array, allow_pickle=False)
    # np.save('./npys/data.npy',out_data,allow_pickle=False)
    np.save('./npys/e1/advdata.npy', out_adv_data, allow_pickle=False)
    # torch.save(out_data,'./npys/data.npy')
    # torch.save(out_adv_data,'./npys/advdata.npy')


def gen_adv_features_train():
    model.eval()
    cnt = 0

    labels = []

    out_data = []
    out_adv_data = []

    out_target = []

    out_output = []
    out_adv_output = []
    for data, target in train_loader:
        cnt += 1
        print("processing: %d/%d" % (cnt, len(train_loader.dataset)))

        #delta = fgsm(model,data,target,0.2)
        #delta = pgd_linf(model,data,target,0.1,1e-2,40)

        output = model(data)
        #adv_output = model(data + delta)

        output_np = output.data.cpu().numpy()
        #adv_output_np = adv_output.data.cpu().numpy()
        # print(np.argmax(output_np),np.argmax(adv_output_np))

        labels.append(target.numpy()[0])
        target_np = output.max(1, keepdim=True)[1][0].numpy()

        #adv_data = data+delta
        ######################

        out_output.append(output_np)
        # out_adv_output.append(adv_output_np)

        out_target.append(target_np[:, np.newaxis])

        data = torch.flatten(data, 2, 3)
        data = torch.flatten(data, 0)
        out_data.append(data.numpy())

        #adv_data = torch.flatten(adv_data,2,3)
        #adv_data = torch.flatten(adv_data,0)
        # out_adv_data.append(adv_data.numpy())

    output_array = np.concatenate(out_output, axis=0)
    #adv_output_array = np.concatenate(out_adv_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)

    np.save('./npys/trainoutput.npy', output_array, allow_pickle=False)
    np.save('./npys/trainlabels.npy', labels, allow_pickle=False)
    #np.save('./npys/advoutput.npy', adv_output_array, allow_pickle=False)
    np.save('./npys/traintarget.npy', target_array, allow_pickle=False)
    np.save('./npys/traindata.npy', out_data, allow_pickle=False)
    # np.save('./npys/advdata.npy',out_adv_data,allow_pickle=False)
    # torch.save(out_data,'./npys/data.npy')
    # torch.save(out_adv_data,'./npys/advdata.npy')



gen_adv_features_train()
gen_adv_features_test()
exit(0)


# Tests neural network on FGSM of various epsilons and saves images
def test(model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []
    examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        # get the index of the max log-probability
        # exit()
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]

        # Save images for examples in gui
        if len(examples) < 500:
            ex = perturbed_data.squeeze().detach().cpu().numpy()
            examples.append(
                (init_pred.item(), final_pred.item(), ex))

        # Check for success
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 50):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 500:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon,
          correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, examples, adv_examples


# Sets model to evaluation and creates arrays for images to be saved to
accuracies = []
examples = []
misclassified_examples = []
model.eval()

# Sets up Epsilons to run through for testing
epsilons = [0, .05, .1, .15, .2, .25, .3]


# Run test for each epsilon
print("Model running through test data...")
for eps in epsilons:
    acc, ex, misEx = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
    misclassified_examples.append(misEx)

# Saves each epsilon to a different variable. This probably should be changed for readability
example0, example0_05, example0_1, example0_15, example0_2, example0_25, example0_3 = examples
misclassified0, misclassified0_05, misclassified0_1, misclassified0_15, misclassified0_2, misclassified0_25, misclassified0_3 = misclassified_examples


images = load('data.npy', allow_pickle=True)
