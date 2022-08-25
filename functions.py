import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tkinter import Scrollbar
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generateEpsilonList(epsilonStepSize,maxEpsilon):
    return [x * epsilonStepSize for x in range(0, math.floor(1+maxEpsilon*(1/epsilonStepSize)))]

class AutoScrollbar(Scrollbar):
    # Defining set method with all its parameters
    def set(self, low, high):
        if float(low) <= 0.0 and float(high) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        Scrollbar.set(self, low, high)

    def pack(self, **kw):
        raise (TclError,"pack cannot be used with this widget")

    def place(self, **kw):
        raise (TclError, "place cannot be used  with this widget")

def SoftCrossEntropyLoss(input, target):
  logprobs = F.log_softmax(input, dim = 1)
  print("1")
  print(logprobs)

  val = -(target * logprobs).sum()
  print("2")
  print(val)
  return val / input.shape[0]

def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = F.nll_loss(model(X+delta), y)
    loss.backward()

    return epsilon * delta.grad.detach().sign()

def pgd(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = F.nll_loss(model(X+delta), y)
        #loss = nn.CrossEntropyLoss()(model(X + delta), y)
        #print(loss)
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = F.nll_loss(model(X+delta), y)
        #loss = nn.CrossEntropyLoss()(model(X + delta), y)
        #print("loss")
        #print(loss)
        loss.backward()
        #print(delta.grad.detach())
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_targ(model, X, y, epsilon, alpha, num_iter, y_targ):
    """ Construct targeted adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        yp = model(X + delta)
        #print(yp)
        loss = (yp[:,y_targ] - yp.gather(1,y[:,None])[:,0]).sum()
        #print(yp[:,y_targ])
        #print(yp.gather(1,y[:,None])[:,0])
        #print(loss)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        #print(delta.data)
        delta.grad.zero_()
    return delta.detach()

def epoch_adversarial(model, loader, attack, *args):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def accuracyfinder(model, loader):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def pgd_l2(model, X, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        #_x_adv = x_adv.clone().detach().requires_grad_(True)
        #loss = F.nll_loss(model(X+delta), y)
        loss = nn.CrossEntropyLoss()(model(X+delta), y)

        loss.backward()
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()

    return delta.detach()
