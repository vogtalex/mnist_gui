import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
from os.path import exists

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def SoftCrossEntropyLoss(input, target):

  logprobs = torch.nn.functional.log_softmax(input, dim = 1)
  print("1")
  print(logprobs)

  val = -(target * logprobs).sum()
  print("2")
  print(val)
  return val / input.shape[0]



def plot_images(X,y,yp,M,N,name):
    print(X.shape())
    print(y.shape())
    print(yp.shape())
    print('hihihih')
    exit()
    plt.clf()
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N,M*1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1-X[i*N+j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("Pred: {}".format(yp[i*N+j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i*N+j].max(dim=0)[1] == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    #plt.savefig(name)
    plt.show()


def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = torch.nn.functional.nll_loss(model(X+delta), y)
    #loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    """
    print(delta.grad.detach().sign().cpu()[0,0,...])
    plt.imshow(delta.grad.detach().sign().cpu()[0,0,...])
    plt.savefig("test")
    exit(0)
   """

    """plt.figure(figsize=(5,5))
    plt.plot(epsilons, b, label='baseline')
    plt.plot(epsilons, s, label='student')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(lower, upper+step, step=step))
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()"""
    return epsilon * delta.grad.detach().sign()


# In[121]:


def pgd(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = torch.nn.functional.nll_loss(model(X+delta), y)
        #loss = nn.CrossEntropyLoss()(model(X + delta), y)
        #print(loss)
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


# In[14]:


def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = torch.nn.functional.nll_loss(model(X+delta), y)
        #loss = nn.CrossEntropyLoss()(model(X + delta), y)
        #print("loss")
        #print(loss)
        loss.backward()
        #print(delta.grad.detach())
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


# In[15]:


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


# In[82]:


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
        #loss = torch.nn.functional.nll_loss(model(X+delta), y)
        loss = nn.CrossEntropyLoss()(model(X+delta), y)

        loss.backward()
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()

    return delta.detach()



def create_graphs(epsilons, student, baseline, test_loader, title, pdf, attack, *args):
    print(f'running {title}')
    upper = max(epsilons)
    lower = min(epsilons)
    step = (upper + lower)/(len(epsilons) - 1)

    students = []
    baselines = []
    for e in epsilons:
        students.append(epoch_adversarial(student, test_loader, attack, e, *args)[0])
        baselines.append(epoch_adversarial(baseline, test_loader, attack, e, *args)[0])

    s = [(1 - a) for a in students]
    b = [(1 - a) for a in baselines]

    plt.figure(figsize=(5,5))
    plt.plot(epsilons, b, label='baseline')
    plt.plot(epsilons, s, label='student')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(lower, upper+step, step=step))
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    print(f'saving {pdf}')
    plt.savefig(pdf)
