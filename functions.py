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

def l2_projection(x, x_init, r):
    if r > 0:
        d = torch.max(l2_norm(x, x_init), torch.tensor(r))
        x = x_init + r * (x - x_init) / d.view(x.shape[0], 1, 1, 1)
    else:
        x = x_init

    return x

def PGD_attack(model, device, loss, x, y, epsilon, niter, stepsize, lpnorm=np.inf, randinit=False, debug=False):
    x_ptb = x.detach().clone()
    model.to(device)

    eps = (torch.ones(x.shape[0])*epsilon).to(device)   # for dimension matching in l2 case
    # set randinit to False for this project, it's mainly used for defense training
    if randinit:
        if lpnorm == np.inf:
            x_ptb = x_ptb + torch.zeros_like(x_ptb).uniform_(-epsilon, epsilon) # random start
            x_ptb = torch.clamp(x_ptb, 0, 1)
        if lpnorm == 2:
            u = torch.normal(0, 1, x.shape, device=x.device)
            norm = torch.norm(u.flatten(start_dim=1), p=2, dim=1)
            r = torch.rand(x.shape[0], device=x.device)
            x_curr = (x + epsilon * u * (r / norm).view(x.shape[0], 1, 1, 1)).detach()
            x_curr = l2_projection(x_curr, x, epsilon)

            # x_ptb = x_ptb + torch.zeros_like(x_ptb).uniform_(-((epsilon/(x.shape[1]*x.shape[2]*x.shape[3]))**0.5),
            #     (epsilon/(x.shape[1]*x.shape[2]*x.shape[3]))**0.5)
    for i in range(niter):
        x_ptb.requires_grad_()
        with torch.enable_grad():
            logits = model(x_ptb)
            loss_val = loss(logits, y)
        grad = torch.autograd.grad(loss_val, [x_ptb])[0]
        if lpnorm == np.inf:
            x_ptb = x_ptb.detach() + stepsize * torch.sign(grad.detach())   # perturbation
            x_ptb = torch.min(torch.max(x_ptb, x - epsilon), x + epsilon)   # linf projection
        if lpnorm == 2:
            # perturbation
            grad_norm = torch.norm(grad.view(x.shape[0], -1), p=2, dim=1)
            grad_norm[grad_norm == 0] = 1
            x_ptb = x_ptb + stepsize * grad / grad_norm.view(x.shape[0],1,1,1)
            total_ptb = x_ptb - x
            ptb_norm = torch.norm(total_ptb.view(x.shape[0], -1), p=2, dim=1)
            denom = torch.max(ptb_norm, eps)
            x_ptb = x.detach().clone() + eps.view(x.shape[0],1,1,1) * total_ptb.detach() / denom.view(x.shape[0],1,1,1)
        x_ptb = torch.clamp(x_ptb, 0, 1)    # pixel value constraint
        if debug:
            print('linf norm:', torch.norm(x_ptb-x, p=np.inf).item()*255)
            print('l1 norm:', torch.norm(x_ptb-x, p=1).item()*255)
    return x_ptb.detach()

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
