# Load Python Libraries
import numpy as np
import os
import gzip, pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from functions import *
import torch
limit = 10000
idx = 10

npys = './npys'
eps = 'e1'
exeps = 'e1'
examples = 'examples'

def get_data(npys,eps,examples):
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

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,eps,examples)

print("advdata ",advdata.shape)
print("testlabels ",testlabels.shape)
print("advoutput ",advoutput.shape)

def findNearest():
    k=10
    print("Index: ",idx)
    example = exdata[idx]
    label = np.argmax(exoutput[idx])
    print("Model prediction: ", label)

    l = advdata - example

    norms = np.linalg.norm(l,axis=1)


    top = np.argpartition(norms,k-1)

    print("True label: ", int(exlabels[idx]))
    print("Nearest 10 labels: ")
    print(top[:k])
    print([(int(testlabels[i])) for i in top[:k]])
    #print("Distance to nearest 10 points: ")
    print([(norms[idx]) for idx in top[1:k]])
    return norms, top[1:k],label,int(exlabels[idx])

norms,idxs,prediction,truelabel = findNearest()
print(norms)

print('data shape: ', traindata.shape)
print('labels shape: ', trainlabels.shape)
print('output shape: ', trainoutput.shape)



#for combining data/advdata
#data = np.append(data, advdata, axis=0)

X_2d = []
if exists(os.path.join(npys,eps,'embedding.npy')):
    X_2d = np.load(os.path.join(npys,eps,'embedding.npy')).astype(np.float64)
else:
    tsne = TSNE(n_components=2, random_state=3,perplexity=100)
    X_2d = tsne.fit_transform(traindata)
    np.save('./embedding.npy', X_2d, allow_pickle=False)

img = exdata[idx].reshape((28,28))
plt.imshow(img, cmap='gray')

###HISTOGRAMS###################
"""EPSILON 0.1"""

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e1',examples)
norms,idxs,prediction,truelabel = findNearest()
fig1, axs1 = plt.subplots(10)

for i in range(10):
    axs1[i].hist(norms[(testlabels[...] == i)], bins=100,range=(8,14),density=True)
    axs1[i].text(13,.25,str(i),ha='center')

title = "Epsilon: 0.1\nModel Prediction: %d\nActual Label: %d\nAverage Distance: %f" % (prediction,truelabel,float(sum(norms))/len(norms))
plt.suptitle(title)

##################
"""EPSILON 0.2"""

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e2',examples)
norms,idxs,prediction,truelabel = findNearest()
fig2, axs2 = plt.subplots(10)

for i in range(10):
    axs2[i].hist(norms[(testlabels[...] == i)], bins=100,range=(8,14),density=True)
    axs2[i].text(13,.25,str(i),ha='center')

title = "Epsilon: 0.2\nModel Prediction: %d\nActual Label: %d\nAverage Distance: %f" % (prediction,truelabel,float(sum(norms))/len(norms))
plt.suptitle(title)

###########
"""EPSILON 0.3"""

trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,'e3',examples)
norms,idxs,prediction,truelabel = findNearest()

fig3, axs3 = plt.subplots(10)
for i in range(10):
    axs3[i].hist(norms[(testlabels[...] == i)], bins=100,range=(8,14),density=True)
    axs3[i].text(13,.25,str(i),ha='center')

title = "Epsilon: 0.3\nModel Prediction: %d\nActual Label: %d\nAverage Distance: %f" % (prediction,truelabel,float(sum(norms))/len(norms))
plt.suptitle(title)

##############

plt.show()


def plot_embedding(labels,target_ids,colors,X_2d,testlabels,norms,idxs):
    fig, (ax1,ax2) = plt.subplots(1,2)

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
               cmap='viridis')

    #plot 10 nearest points
    cb = ax2.scatter(X_2d[idxs,0],X_2d[idxs,1],
               c='black',
               label="nearest",
               s=10,
               picker=True)

    title = "Model Prediction: %d\nActual Label: %d\nAverage Distance: %f" % (prediction,truelabel,float(sum(norms))/len(norms))
    ax2.set_title(title)

    plt.colorbar(cb,label="norm")
    cb.set_clim(5,15)
    
    ax1.legend()
    plt.show()


labels = list(range(0, 10))
target_ids = range(10)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'
#plot_embedding(labels,target_ids,colors,X_2d,testlabels,norms,idxs)

print('max distance', max(norms))
print('min distance', min(norms))
print('avg distance', sum(norms)/len(norms))







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
