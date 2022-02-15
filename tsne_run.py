# Load Python Libraries
import numpy as np
import gzip, pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from functions import *
import torch
limit = 60000
idx = 7

trainlabels = np.load('./npys/trainlabels.npy').astype(np.float64)[:limit]
testlabels = np.load('./npys/testlabels.npy').astype(np.float64)[:limit]
trainoutput = np.load('./npys/trainoutput.npy').astype(np.float64)[:limit]
traindata = np.load('./npys/traindata.npy').astype(np.float64)[:limit]
#traintarget = np.load('./npys/traintarget.npy')[:limit]

advoutput = np.load('./npys/advoutput.npy').astype(np.float64)[:limit]
advdata = np.load('./npys/advdata.npy').astype(np.float64)[:limit]


def findNearest(data,advdata,idx,k=10):
    print("Index: ",idx)
    example = advdata[idx]
    label = np.argmax(advoutput[idx])
    print("Model prediction: ", label)

    l = traindata - example

    norms = np.linalg.norm(l,axis=1)


    top = np.argpartition(norms,k-1)

    ########
    print("True label: ", int(testlabels[idx]))
    print("Nearest 10 labels: ")
    print(top[:k])
    print([(int(trainlabels[i])) for i in top[:k]])
    #print("Distance to nearest 10 points: ")
    print([(norms[idx]) for idx in top[1:k]])
    return norms, top[1:k],label,int(testlabels[idx])

def findNearestTrue(data,advdata,idx,k=10):
    print("Index: ",idx)
    example = traindata[idx]
    label = np.argmax(trainoutput[idx])
    print("Model prediction: ", label)

    l = traindata - example

    norms = np.linalg.norm(l,axis=1)


    top = np.argpartition(norms,k-1)

    ########
    print("True label: ", int(trainlabels[idx]))
    print("Nearest 10 labels: ")
    print([(int(trainlabels[idx])) for idx in top[:k]])
    #print("Distance to nearest 10 points: ")
    print([(norms[idx]) for idx in top[1:k]])
    return norms, top[1:k],label,int(trainlabels[idx])

norms,idxs,prediction,truelabel = findNearestTrue(traindata,advdata,idx)
#norms,idxs,prediction,truelabel = findNearest(traindata,advdata,idx)

print('data shape: ', traindata.shape)
print('labels shape: ', trainlabels.shape)
print('output shape: ', trainoutput.shape)
#print(data[0])

fig, (ax1,ax2) = plt.subplots(1,2)

print(traindata.shape)
print(advdata.shape)
#for combining data/advdata
#data = np.append(data, advdata, axis=0)

print(traindata.shape)

X_2d = []
if exists("./npys/embedding.npy"):
    X_2d = np.load('./npys/embedding.npy').astype(np.float64)
else:
    tsne = TSNE(n_components=2, random_state=3,perplexity=100)
    X_2d = tsne.fit_transform(traindata)
    np.save('./npys/embedding.npy', X_2d, allow_pickle=False)

labels = list(range(0, 10))
target_ids = range(10)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'

#plt.scatter(X_2d[:limit, 0], X_2d[:limit, 1], s=5, c='r', label='normal')
#plt.scatter(X_2d[limit:, 0], X_2d[limit:, 1], s=2, c='b', label='attacked')

#unatt = X_2d[:limit]
#att = X_2d[limit:]
#unatt = X_2d

#label for class
for i, c, label in zip(target_ids, colors, labels):
    ax1.scatter(X_2d[(trainlabels[...] == i), 0],
               X_2d[(trainlabels[...] == i), 1],
               c=c,
               label=label,
               s=3,
               picker=True)

ax1.set_title("Test Data TSNE Embedding")

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
ax2.scatter(X_2d[..., 0],
           X_2d[..., 1],
           c=norms[...],
           s=3,
           cmap='viridis')

#ax.scatter(att[..., 0],
#           att[..., 1],
#           c='fuchsia',
#           label="attacked",
#           s=3,
#           picker=True)


cb = ax2.scatter(X_2d[idxs,0],X_2d[idxs,1],
           c='black',
           label="nearest",
           s=10,
           picker=True)

title = "Model Prediction: %d, Actual Label: %d" % (prediction,truelabel)
ax2.set_title(title)
'''

ax2.scatter(adv[idx,0],att[idx,1],
           c='red',
           label="attacked",
           s=10,
           picker=True)
'''

#p=ax2.get_children()[2]
plt.colorbar(cb,label="norm")
#plt.clim(np.argmin(norms),np.argmax(norms))

#ax.scatter(output_2d[:, 0], output_2d[:, 1], c=target)
ax1.legend()
plt.show()
exit(0)

"""
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


def onpick(event):
    mouseevent = event.mouseevent
    thispoint = event.artist
    print("bloop")
    of = thispoint.get_offsets()[0]
    print(of)
    #idx = list(X_2d).index(of.all())
    idx = np.argwhere(X_2d == of)[0][0]
    print(data.shape)
    np.expand_dims(data,axis=(0,1))
    print(data.shape)
    exit()
    plot_images(data[idx],target[idx][0],output[idx],3,6,"deleteme.png")


labels = list(range(0, 10))
target_ids = range(10)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'
unattacked = X_2d[:limit]
attacked = X_2d[limit:]


pairs = zip(xs,zs)

for i, c, label in zip(target_ids, colors, labels):
    ax.scatter(X_2d[(target[...,0] == i), 0],
               X_2d[(target[...,0] == i), 1],
               c=c,
               label=label,
               s=3,
               picker=True)


plt.legend()
plt.savefig('./coloredavn.png', bbox_inches='tight')
#fig.canvas.mpl_connect('button_press_event',onclick)
fig.canvas.mpl_connect('pick_event',onpick)
#fig.canvas.callbacks.connect('pick_event',on_pick)
plt.show()
exit()

'''
for i, c, label in zip(target_ids, colors, labels):
    xs = X_2d[100:]
    plt.scatter(xs[(advtarget[...,0] == i), 0], xs[(advtarget[...,0] == i), 1], s=3, c=c)

'''
#plt.scatter(X_2d[:10000, 0], X_2d[:10000, 1], s=5, c='r', label='normal')
#plt.scatter(X_2d[10000:, 0], X_2d[10000:, 1], s=2, c='b', label='attacked')
plt.legend()
plt.savefig('./coloredavn.png', bbox_inches='tight')
plt.show()
exit()

labels = list(range(0, 10))
target_ids = range(10)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'

tsne = TSNE(n_components=2, random_state=1)
X_2d = tsne.fit_transform(output)

for i, c, label in zip(target_ids, colors, labels):
    plt.scatter(X_2d[target[...,0] == i, 0], X_2d[target[...,0] == i, 1], s=10, c=c, label=label)
plt.legend()
plt.savefig('./x2d.png', bbox_inches='tight')
plt.show()

exit()

plt.scatter(output_2d[:, 0], output_2d[:, 1], c=target)
target_ids = range(len(data.target_names))


exit()

output_2d = bh_sne(output)

np.save('./output_2d.npy', output_2d, allow_pickle=False)


plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(output_2d[:, 0], output_2d[:, 1], c=target)
plt.legend()
plt.savefig('./output_2d.png', bbox_inches='tight')
plt.show()

"""
