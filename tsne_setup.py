import os
os.environ['OPENBLAS_NUM_THREADS']='5'
import torch
from functions import *
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import json
import sys

with open('config.json') as f:
   config = json.load(f)

# split file into name and path
head,tail = os.path.split(config['Model']['modelDir'])
# temporarily add directory where python file is present to path
sys.path.append(head)
# format import with specific module name and execute the import
exec(f"from {tail[:-3]} import *")

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1000, shuffle=False)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1000, shuffle=True)
# Starter code if using cuda
use_cuda = True
device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Use a pretrained model
pretrained_model = config["Model"]["weightDir"]

# Initialize the network
model = Net()
model.load_state_dict(torch.load(pretrained_model,map_location=device))

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


# def gen_adv_features_train():
#     model.eval()
#     cnt = 0
#
#     labels = []
#
#     out_data = np.array([])
#     out_adv_data= []
#
#     out_target = []
#
#     out_output = np.array([])
#     out_adv_output = []
#     for data, target in train_loader:
#         data,target = data.to(device), target.to(device)
#         cnt += 1
#         print("processing: %d/%d" % (cnt, len(train_loader.dataset)))
#
#         #delta = fgsm(model,data,target,0.2)
#         #delta = pgd_linf(model,data,target,0.1,1e-2,40)
#
#         output = model(data)
#         #adv_output = model(data + delta)
#         output_np = output.data.cpu().numpy()
#         out_output = np.vstack([out_output,output_np]) if out_output.size else output_np
#         #adv_output_np = adv_output.data.cpu().numpy()
#         #print(np.argmax(output_np),np.argmax(adv_output_np))
#
#
#         #print(labels)
#         labels = np.append(labels,target.cpu().numpy())
#
#         #adv_data = data+delta
#         ######################
#
#         #out_adv_output.append(adv_output_np)
#
#
#         data = torch.flatten(data,2,3)
#         data = torch.flatten(data,0,1)
#         out_data = np.vstack([out_data,data.cpu().numpy()]) if out_data.size else data.cpu().numpy()
#         #adv_data = torch.flatten(adv_data,2,3)
#         #adv_data = torch.flatten(adv_data,0)
#         #out_adv_data.append(adv_data.numpy())
#
#
#     #output_array = np.concatenate(out_output, axis=0)
#     #adv_output_array = np.concatenate(out_adv_output, axis=0)
#     #target_array = np.concatenate(out_target, axis=0)
#
#     np.save('./npys/trainoutput.npy', out_output, allow_pickle=False)
#     np.save('./npys/trainlabels.npy', labels, allow_pickle=False)
#     #np.save('./npys/advoutput.npy', adv_output_array, allow_pickle=False)
#     #np.save('./npys/traintarget.npy', target_array, allow_pickle=False)
#     np.save('./npys/traindata.npy',out_data,allow_pickle=False)
#     #np.save('./npys/advdata.npy',out_adv_data,allow_pickle=False)
#     #torch.save(out_data,'./npys/data.npy')
#     #torch.save(out_adv_data,'./npys/advdata.npy')

def gen_adv_features_test():
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
        print("processing: %d/%d" % (cnt, len(train_loader.dataset)))
        if cnt > 9:
            break;

        #delta = fgsm(model,data,target,0.2)
        #delta = pgd_linf(model,data,target,0.2,1e-2,40)
        #data = data+delta

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
        data = torch.flatten(data,0,1)
        out_data = np.vstack([out_data,data.cpu().numpy()]) if out_data.size else data.cpu().numpy()
        #adv_data = torch.flatten(adv_data,2,3)
        #adv_data = torch.flatten(adv_data,0)
        #out_adv_data.append(adv_data.numpy())


    np.save('./npys/e0/testlabels.npy', labels, allow_pickle=False)

    np.save('./npys/e0/advoutput.npy', out_output, allow_pickle=False)
    np.save('./npys/e0/advdata.npy',out_data,allow_pickle=False)

def gen_adv_features_examples():
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
        print("processing: %d/%d" % (cnt, len(train_loader.dataset)))
        if cnt <= 9:
            continue;

        #delta = fgsm(model,data,target,0.2)
        delta = pgd_linf(model,data,target,0.3,1e-2,100)

        output = model(data+delta)
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


        data=data+delta
        data = torch.flatten(data,2,3)
        data = torch.flatten(data,0,1)
        out_data = np.vstack([out_data,data.cpu().numpy()]) if out_data.size else data.cpu().numpy()
        #adv_data = torch.flatten(adv_data,2,3)
        #adv_data = torch.flatten(adv_data,0)
        #out_adv_data.append(adv_data.numpy())


    np.save('./npys/examples/e1/testlabels.npy', labels, allow_pickle=False)

    np.save('./npys/examples/e1/advoutput.npy', out_output, allow_pickle=False)
    np.save('./npys/examples/e1/advdata.npy',out_data,allow_pickle=False)

# gen_adv_features_examples()
gen_adv_features_test()
gen_adv_features_train()
