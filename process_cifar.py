
import os
os.environ['OPENBLAS_NUM_THREADS']='5'
import torch
from functions import *
from resnet_cifar10.resnet import *
#from madrynet.madrynet import *
import numpy as np
#import matplotlib.pyplot as plt

import time
import tkinter as tk

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

#import tkinter as tk
#from tkinter import *
#
#from PIL import Image, ImageTk

from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"]="0"


batch_size=100


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./datacifar',train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./datacifar',train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Starter code if using cuda
use_cuda = True
#device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")

device = torch.device("cuda:0")

# Use a pretrained model
pretrained_model = "./resnet_cifar10/resnet18_cifar10_l2_1.6.pth"

# Initialize the network
#model = ResNet()
#model = ResNet18().to(device)
model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
model.load_state_dict(torch.load(pretrained_model,map_location=device))

# Creates a random seed
random_seed = 1
torch.backends.cudnn.enabled = True
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


def gen_adv_features_train(eps,loc,attstrength,iters,stpsz):
    model.eval()
    cnt = 0

    labels = []

    out_data = np.array([])
    out_adv_data= []

    out_target = []

    out_output = np.array([])
    out_adv_output = []
    for data, target in train_loader:
        data,target = data.to(device), target.to(device)
        cnt += 1
        print("processing: %d/%d" % (cnt, len(train_loader.dataset)))

        #delta = fgsm(model,data,target,0.2)
        #delta = pgd_linf(model,data,target,0.1,1e-2,40)

        output = model(data)
        #adv_output = model(data + delta)
        output_np = output.data.cpu().numpy()
        out_output = np.vstack([out_output,output_np]) if out_output.size else output_np
        #adv_output_np = adv_output.data.cpu().numpy()
        #print(np.argmax(output_np),np.argmax(adv_output_np))


        #print(labels)
        labels = np.append(labels,target.cpu().numpy())

        #adv_data = data+delta
        ######################

        #out_adv_output.append(adv_output_np)


        data = torch.flatten(data,2,3)
        data = torch.flatten(data,1,2)
        out_data = np.vstack([out_data,data.cpu().numpy()]) if out_data.size else data.cpu().numpy()
        #adv_data = torch.flatten(adv_data,2,3)
        #adv_data = torch.flatten(adv_data,0)
        #out_adv_data.append(adv_data.numpy())

    np.save(os.path.join(loc,'train','trainoutput.npy'), out_output, allow_pickle=False)
    np.save(os.path.join(loc,'train','trainlabels.npy'), labels, allow_pickle=False)
    np.save(os.path.join(loc,'train','traindata.npy'), out_data, allow_pickle=False)


def gen_adv_features_test(eps,loc,attstrength,iters,stpsz):
    model.eval()
    cnt = 0

    labels = []

    out_data = np.array([])
    out_adv_data= []

    out_target = []

    out_output = np.array([])
    out_adv_output = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        cnt += 1
        print("processing: %d/%d" % (cnt, len(test_loader.dataset)/batch_size))
        if cnt > 90:
            break

        #delta = fgsm(model,data,target,0.2)
        #delta = pgd_linf(model,data,target,0.3,1e-2,100)
        #delta = pgd_linf_rand(model,data,target,0.3,1e-2,100,2)
        delta,loss_arr = pgd_l2(model, data, target, epsilon=attstrength, alpha=stpsz, num_iter=iters)
        data = data + delta
        data = torch.clamp(data,0,1)

        output = model(data)
        #adv_output = model(data + delta)
        output_np = output.data.cpu().numpy()
        out_output = np.vstack([out_output,output_np]) if out_output.size else output_np
        #adv_output_np = adv_output.data.cpu().numpy()
        #print(np.argmax(output_np),np.argmax(adv_output_np))


        #print(labels)
        labels = np.append(labels,target.cpu().numpy())

        #adv_data = data+delta
        ######################

        #out_adv_output.append(adv_output_np)
        data = torch.flatten(data,2,3)
        data = torch.flatten(data,1,2)
        out_data = np.vstack([out_data,data.cpu().numpy()]) if out_data.size else data.cpu().numpy()
        #adv_data = torch.flatten(adv_data,2,3)
        #adv_data = torch.flatten(adv_data,0)
        #out_adv_data.append(adv_data.numpy())

    np.save(os.path.join(loc,'test',eps,'testlabels.npy'), labels, allow_pickle=False)
    np.save(os.path.join(loc,'test',eps,'advoutput.npy'), out_output, allow_pickle=False)
    np.save(os.path.join(loc,'test',eps,'advdata.npy'), out_data, allow_pickle=False)

    #np.save('./madrys2/e0/testlabels.npy', labels, allow_pickle=False)
    #np.save('./madrys2/e0/advoutput.npy', out_output, allow_pickle=False)
    #np.save('./madrys2/e0/advdata.npy',out_data,allow_pickle=False)

def gen_adv_features_examples(eps,loc,attstrength,iters,stpsz):
    model.eval()
    cnt = 0

    labels = []

    out_data = np.array([])
    out_adv_data= []

    out_target = []

    out_output = np.array([])
    out_adv_output = []
    for data, target in test_loader:
        data,target = data.to(device), target.to(device)
        cnt += 1
        print("processing: %d/%d" % (cnt, len(test_loader.dataset)/batch_size))
        if cnt <= 90:
            continue;

        #delta = fgsm(model,data,target,0.2)
        #delta = pgd_linf(model,data,target,0.1,1e-2,100)
        #delta = pgd_linf_rand(model,data,target,0.3,1e-2,100,2)
        delta,loss_arr = pgd_l2(model, data, target, epsilon=attstrength, alpha=stpsz, num_iter=iters)
        data = data+delta
        data = torch.clamp(data,0,1)

        output = model(data)
        #adv_output = model(data + delta)
        output_np = output.data.cpu().numpy()
        out_output = np.vstack([out_output,output_np]) if out_output.size else output_np
        #adv_output_np = adv_output.data.cpu().numpy()
        #print(np.argmax(output_np),np.argmax(adv_output_np))


        #print(labels)
        labels = np.append(labels,target.cpu().numpy())

        #adv_data = data+delta
        ######################

        #out_adv_output.append(adv_output_np)


        data = torch.flatten(data,2,3)
        data = torch.flatten(data,1,2)
        out_data = np.vstack([out_data,data.cpu().numpy()]) if out_data.size else data.cpu().numpy()
        #adv_data = torch.flatten(adv_data,2,3)
        #adv_data = torch.flatten(adv_data,0)
        #out_adv_data.append(adv_data.numpy())


    #np.save('./madrys2/examples/e0/testlabels.npy', labels, allow_pickle=False)
    #np.save('./madrys2/examples/e0/advoutput.npy', out_output, allow_pickle=False)
    #np.save('./madrys2/examples/e0/advdata.npy',out_data,allow_pickle=False)
    np.save(os.path.join(loc,'examples',eps,'testlabels.npy'), labels, allow_pickle=False)
    np.save(os.path.join(loc,'examples',eps,'advoutput.npy'), out_output, allow_pickle=False)
    np.save(os.path.join(loc,'examples',eps,'advdata.npy'), out_data, allow_pickle=False)

#'''
epslist = ['e1','e2','e3','e4']
attstrengthlist = [1,3,5,7]
loc = './cifar_npys'

for i in range(4):
    eps = epslist[i]
    attstrength = attstrengthlist[i]
    stpsz = 2.5*attstrength/100
    #stpsz = 5*attstrength/100

    iters = 200
    print(eps,loc,attstrength,iters,stpsz)

    print("test")
    gen_adv_features_test(eps,loc,attstrength,iters,stpsz)
    print("examples")
    gen_adv_features_examples(eps,loc,attstrength,iters,stpsz)

#'''
#eps = 'e0'
#loc = './cifar_npys'
#attstrength = 20
#iters = 100
#stpsz = 2.5*attstrength/iters
##gen_adv_features_train(eps,loc,attstrength,iters,stpsz)
#gen_adv_features_test(eps,loc,attstrength,iters,stpsz)
#gen_adv_features_examples(eps,loc,attstrength,iters,stpsz)
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
        #exit()
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


# Initializes an image for GUI
label, new_label, image, = example0_15[0]
plt.title("What is this number?")
plt.imshow(image, cmap="gray")
plt.savefig('saved_figure.png')


# Generates an image of epsilon 0.15
def generateNewImage(count):
    label, new_label, image, = example0_15[count]
    plt.title("What is this number?")
    plt.imshow(image, cmap="gray")
    plt.savefig('saved_figure2.png')
    return label



# Generates an image of epsilon 0.15
def generateNewMisclassifiedImage(count):
    label, new_label, image, = misclassified0_15[count]
    plt.title("What is this number?")
    plt.imshow(image, cmap="gray")
    plt.savefig('saved_figure2.png')
    return label



# Initializes the height and width of image for GUI
HEIGHT = 200
WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200

# Variables initialized for my user path. Can be changed for different user
path1 = "saved_figure.png"
path2 = "saved_figure2.png"
path3 = "saved_figure3.png"


# Count of image displayed currently and count guessed correctly in the GUI
correctCount = 0
totalCount = 1

# Iterates the total count to iterate through images
def countIterator():
    global totalCount
    totalCount = totalCount + 1
    return totalCount

def quitFunction():
    root.destroy()
    quit()

# Function to display perturbed images for user based on user input
def numClick():
    currNum = a.get()
    temp = 0
    # curr = len(d) - 1
    curr = 499
    images = []
    while temp < 6 and curr > 0:
        label, new_label, image = misclassified0_15[curr]
        if label == int(currNum):
            images.append(image)
            temp = temp + 1
        curr = curr - 1

    for currIterator in range(6):
        currImage = images[currIterator]
        plt.imshow(currImage, cmap="gray")
        plt.savefig('saved_figure3.png')

        image1 = Image.open(path3)
        test = ImageTk.PhotoImage(image1, master=root)
        label1 = tk.Label(number_frame, image=test)
        label1.image = test
        if currIterator < 3:
            label1.grid(row=3, column=currIterator,
                        sticky="nsew", padx=2, pady=2)
        else:
            label1.grid(row=4, column=(currIterator - 3),
                        sticky="nsew", padx=2, pady=2)


# Function for button for user guess
def myClick():
    totalCount = countIterator()
    global label
    currCount = correctCount
    newLabel = label
    currNum = e.get()

    if (int(currNum) == newLabel):
        currCount = currCount + 1
        myLabel = Label(output_frame, text="Correct!")
        myLabel.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=2, pady=2)
    else:
        myLabel2 = Label(output_frame, text="Incorrect")
        myLabel2.grid(row=0, column=2, rowspan=2,
                      sticky="nsew", padx=2, pady=2)

    # Create new image
    label = generateNewImage(totalCount)

    image1 = Image.open(path2)
    test = ImageTk.PhotoImage(image1, master=root)
    label1 = tk.Label(image_frame, image=test)
    label1.image = test
    label1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

    # Generates new model prediction
    stringModel = "Model Prediction: "
    ga, answer, gar, = example0_15[totalCount]
    convAnswer = str(answer)
    stringModel = stringModel + convAnswer
    def_label = tk.Label(visual_aid_frame, text=stringModel)
    def_label.pack(padx=10, pady=5, fill=tk.BOTH)

    #Label(image_frame, image=test).grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
    totalCount = totalCount + 1




# GUI
root = Tk()
root.title("Human Testing of Adversarial Training")

# Setup frames
# global image_frame
image_frame = tk.Frame(root, background="#FFF0C1", bd=1, relief="sunken")
input_frame = tk.Frame(root, background="#D2E2FB", bd=1, relief="sunken")
visual_aid_frame = tk.Frame(root, background="#CCE4CA", bd=1, relief="sunken")
output_frame = tk.Frame(root, background="#F5C2C1", bd=1, relief="sunken")
number_frame = tk.Frame(root, background="#FFF0C1", bd=1, relief="sunken")
image_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
input_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
visual_aid_frame.grid(row=0, column=1, rowspan=2,
                      sticky="nsew", padx=2, pady=2)
output_frame.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=2, pady=2)
number_frame.grid(row=0, column=3, rowspan=2, sticky="nsew", padx=2, pady=2)


# Configure frames
root.grid_rowconfigure(0, weight=3)
root.grid_rowconfigure(1, weight=2)
root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=2)
root.grid_columnconfigure(2, weight=2)
root.grid_columnconfigure(3, weight=2)


# Create a photoimage object of the image in the path
image1 = Image.open(path1)
test = ImageTk.PhotoImage(image1, master=root)
Label(image_frame, image=test).grid(
    row=0, column=0, sticky="nsew", padx=2, pady=2)

# Creates entry box for user guess
e = Entry(input_frame, width=50)
e.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)


# Adds a Button
myButton = Button(input_frame, text="Click Me!",
                  pady=50, command=partial(myClick))
myButton.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

# Visual Aid
stringModel = "Model Prediction: "
ga, answer, gar, = misclassified0_15[totalCount]
convAnswer = str(answer)
stringModel = stringModel + convAnswer
def_label = tk.Label(visual_aid_frame, text=stringModel)
def_label.pack(padx=10, pady=5, fill=tk.BOTH)

# Show number based on user input
num_label = tk.Label(
    number_frame, text="What number would you like to see perturbed images of?")
num_label.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

a = Entry(number_frame, width=50)
a.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

myButton = Button(number_frame, text="Click Me!",
                  pady=50, command=partial(numClick))
myButton.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)

exitButton = Button(number_frame, text="Quit",
                  pady=50, command=quitFunction)
exitButton.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

# Position image
root.mainloop()
