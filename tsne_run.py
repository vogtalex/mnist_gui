# Load Python Libraries
import numpy as np
import os
import gzip, pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.manifold import TSNE
from functions import *
import torch
limit = 9000
idx = 3
#idx 41
#idx 48

npys = './madrys2'
#what epsilon for embedding
eps = 'e3'
#what attack level of example point
exeps ='e3'
examples = 'examples'
eps_dict = {'e0':'Epsilon 0.0', 'e1':'Epsilon 2', 'e2': 'Epsilon 4', 'e3':'Epsilon 6', 'e4':'Epsilon 10'}

def get_data(npys,eps,examples,exeps):
    #train data
    trainlabels = np.load(os.path.join(npys,'trainlabels.npy')).astype(np.float64)[:limit]
    trainoutput = np.load(os.path.join(npys,'trainoutput.npy')).astype(np.float64)[:limit]
    traindata = np.load(os.path.join(npys,'traindata.npy')).astype(np.float64)[:limit]
    
    #adversarial data
    testlabels = np.load(os.path.join(npys,eps,'testlabels.npy')).astype(np.float64)[:limit]
    advoutput = np.load(os.path.join(npys,eps,'advoutput.npy')).astype(np.float64)[:limit]
    advdata = np.load(os.path.join(npys,eps,'advdata.npy')).astype(np.float64)[:limit]
    
    #example data
    exlabels = np.load(os.path.join(npys,examples,exeps,'testlabels.npy')).astype(np.float64)[:limit]
    exoutput = np.load(os.path.join(npys,examples,exeps,'advoutput.npy')).astype(np.float64)[:limit]
    exdata = np.load(os.path.join(npys,examples,exeps,'advdata.npy')).astype(np.float64)[:limit]
    return trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,eps,examples,exeps)

#print("advdata ",advdata.shape)
#print("exdata ",exdata.shape)
#print("testlabels ",testlabels.shape)
#print("advoutput ",advoutput.shape)

def findNearest(exdata,exoutput,exlabels,advdata,testlabels):
    k=10
    print("Index: ",idx)
    example = exdata[idx]
    label = np.argmax(exoutput[idx])
    print("Model prediction: ", label)

    l = advdata - example

    norms = np.linalg.norm(l, axis=1)
    #norms = np.linalg.norm(l,ord=np.inf, axis=1)


    top = np.argpartition(norms,k-1)

    print("True label: ", int(exlabels[idx]))
    #print("Nearest 10 labels: ")
    #print(top[:k])
    print([(int(testlabels[i])) for i in top[:k]])
    #print("Distance to nearest 10 points: ")
    print([(norms[idx]) for idx in top[1:k]])
    return norms, top[1:k],label,int(exlabels[idx])

#norms,idxs,prediction,truelabel = findNearest()
norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels)

exdata1 = np.load(os.path.join(npys,examples,'e0','advdata.npy')).astype(np.float64)[:limit]
img = exdata1[idx].reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

exdata2 = np.load(os.path.join(npys,examples,exeps,'advdata.npy')).astype(np.float64)[:limit]
img = exdata2[idx].reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

print(np.linalg.norm(exdata1[idx] - exdata2[idx]))

print('max distance', max(norms))
print('min distance', min(norms))
print('avg distance', sum(norms)/len(norms))
#print('data shape: ', traindata.shape)
#print('labels shape: ', trainlabels.shape)
#print('output shape: ', trainoutput.shape)



#for combining data/advdata
#data = np.append(data, advdata, axis=0)

'''
X_2d = []
if exists(os.path.join(npys,eps,'embedding.npy')):
    X_2d = np.load(os.path.join(npys,eps,'embedding.npy')).astype(np.float64)
else:
    tsne = TSNE(n_components=2, random_state=4,perplexity=100)
    X_2d = tsne.fit_transform(advdata)
    np.save('./embedding.npy', X_2d, allow_pickle=False)
'''



def plot_embedding(labels,target_ids,colors,X_2d,testlabels,norms,idxs):
    fig, (ax1,ax2) = plt.subplots(1,2)
    normalized = mpl.colors.Normalize(5,17)

    #plot embedding for class coloration
    for i, c, label in zip(target_ids, colors, labels):
        ax1.scatter(X_2d[(testlabels[...] == i), 0],
                   X_2d[(testlabels[...] == i), 1],
                   c=c,
                   label=label,
                   s=3,
                   picker=True)

    ax1.set_title("Test Data")

    #plot embedding for norm coloration
    ax2.scatter(X_2d[..., 0],
               X_2d[..., 1],
               c=norms[...],
               s=3,
               cmap='viridis', norm=normalized)

    #plot 10 nearest points
    cb = ax2.scatter(X_2d[idxs,0],X_2d[idxs,1],
               c='red',
               label="nearest",
               s=10,
               picker=True)

    title = "Model Prediction: %d\nActual Label: %d\nAverage Distance: %f" % (prediction,truelabel,float(sum(norms))/len(norms))
    ax2.set_title(title)

    plt.colorbar(cb,label="norm")
    cb.set_clim(5,17)
    
    ax1.legend()
    plt.show()


labels = list(range(0, 10))
target_ids = range(10)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'
#plot_embedding(labels,target_ids,colors,X_2d,testlabels,norms,idxs)


###HISTOGRAMS###################

b=None
r=None
#r=(0.99,1.01)
r=(5,16)
b=200

fig, axs = plt.subplots(10)
fig2, axs2 = plt.subplots()
norm_list = []

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e0',examples,exeps)
norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels)

for i in range(10):
    axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='unattacked', histtype="step")
    axs[i].text(13,.25,str(i),ha='center')
    
norm_list.append(norms)
#axs2.boxplot(norms, patch_artist = True,notch ='True', vert = 0)


"""EPSILON 0.1"""


trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e1',examples,exeps)
norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels)
#trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, _,_,_ = get_data(npys,'e1',examples,exeps)
#norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels)

for i in range(10):
    axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 2', histtype="step")
norm_list.append(norms)


##################
"""EPSILON 0.2"""


trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e2',examples,exeps)
#trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, _,_,_ = get_data(npys,'e2',examples,exeps)
norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels)

for i in range(10):
    axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 4', histtype="step")

norm_list.append(norms)


###########
"""EPSILON 0.3"""


trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e3',examples,exeps)
#trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, _,_,_ = get_data(npys,'e3',examples,exeps)
norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels)

for i in range(10):
    axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 6', histtype="step")
norm_list.append(norms)

############EPSILON 4#################
trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e4',examples,exeps)
#trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, _,_,_ = get_data(npys,'e4',examples,exeps)
norms,idxs,prediction,truelabel = findNearest(exdata,exoutput,exlabels,advdata,testlabels)

for i in range(10):
    axs[i].hist(norms[(testlabels[...] == i)], alpha=0.5, bins=b,range=r,density=True, label='Epsilon 10', histtype="step")

norm_list.append(norms)
'''
fig, ax = plt.subplots()
num = 6

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e0',examples)
norms,idxs,prediction,truelabel = findNearest()
ax.hist(norms[(testlabels[...] == num)], alpha=0.5, bins=b,range=r,density=True, label='eps 0.0')

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e1',examples)
norms,idxs,prediction,truelabel = findNearest()
ax.hist(norms[(testlabels[...] == num)], alpha=0.5, bins=b,range=r,density=True, label='eps 0.1')

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e2',examples)
norms,idxs,prediction,truelabel = findNearest()
ax.hist(norms[(testlabels[...] == num)], alpha=0.5, bins=b,range=r,density=True, label='eps 0.2')

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e3',examples)
norms,idxs,prediction,truelabel = findNearest()
ax.hist(norms[(testlabels[...] == num)], alpha=0.5, bins=b,range=r,density=True, label='eps 0.3')
'''




##############
#title = "%s\nModel Prediction: %d\nActual Label: %d" % (eps_dict[exeps],prediction,truelabel)
title = "Model Prediction: %d" % (prediction)
axs2.boxplot(norm_list, patch_artist = True,notch ='True', vert = 1,labels=['unattacked','Epsilon 2', 'Epsilon 4', 'Epsilon 6', 'Epsilon 10'], showmeans=True)
plt.suptitle(title)
plt.legend(loc='upper left')
plt.show()






#####OLD FUNCTIONS BELOW############


#plt.scatter(X_2d[:limit, 0], X_2d[:limit, 1], s=5, c='r', label='normal')
#plt.scatter(X_2d[limit:, 0], X_2d[limit:, 1], s=2, c='b', label='attacked')

#label for class
'''
for i, c, label in zip(target_ids, colors, labels):
    ax1.scatter(X_2d[(trainlabels[...] == i), 0],
               X_2d[(trainlabels[...] == i), 1],
               c=c,
               label=label,
               s=3,
               picker=True)
'''


#label for norms
"""
for i, c, label in zip(target_ids, colors, labels):
    ax2.scatter(X_2d[(trainlabels == i), 0],
               X_2d[(trainlabels == i), 1],
               c=norms[(trainlabels == i)],
               s=3,
               picker=True,
               cmap='viridis')
"""

#for i, c, label in zip(target_ids, colors, labels):

#ax.scatter(att[..., 0],
#           att[..., 1],
#           c='fuchsia',
#           label="attacked",
#           s=3,
#           picker=True)





'''
ax2.scatter(adv[idx,0],att[idx,1],
           c='red',
           label="attacked",
           s=10,
           picker=True)
'''

#p=ax2.get_children()[2]


def onpick(event):
    mouseevent = event.mouseevent
    thispoint = event.artist
    print("bloop")
    of = thispoint.get_offsets()[0]
    print(of)
    idx = np.argwhere(X_2d == of)[0][0]
    print(data.shape)
    np.expand_dims(data,axis=(0,1))
    print(data.shape)
    exit()
    plot_images(data[idx],target[idx][0],output[idx],3,6,"deleteme.png")
