# Load Python Libraries
import numpy as np
import gzip, pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from functions import *
import torch
limit = 100

labelling = np.load('./npys/labels.npy').astype(np.float64)[:limit]
output = np.load('./npys/output.npy').astype(np.float64)[:limit]
data = torch.load('./npys/data.npy')
data = torch.flatten(data,2,3)
data = torch.flatten(data,1)
data = data.numpy()[:limit]
target = np.load('./npys/target.npy')[:limit]

advoutput = np.load('./npys/advoutput.npy').astype(np.float64)[:limit]

advdata = torch.load('./npys/advdata.npy')
advdata = torch.flatten(advdata,2,3)
advdata = torch.flatten(advdata,1)
advdata = advdata.numpy()[:limit]

print('data shape: ', data.shape)
print('target shape: ', target.shape)
print('output shape: ', output.shape)
#print(data[0])

fig, ax = plt.subplots()

#for combining data/advdata
data = np.append(data, advdata, axis=0)

tsne = TSNE(n_components=2, random_state=3)
X_2d = tsne.fit_transform(data)

labels = list(range(0, 10))
target_ids = range(10)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'

#plt.scatter(X_2d[:limit, 0], X_2d[:limit, 1], s=5, c='r', label='normal')
#plt.scatter(X_2d[limit:, 0], X_2d[limit:, 1], s=2, c='b', label='attacked')

unatt = X_2d[:100]
att = X_2d[100:]

ax.scatter(att[..., 0],
           att[..., 1],
           c='fuchsia',
           label="attacked",
           s=5,
           picker=True)

for i, c, label in zip(target_ids, colors, labels):

    ax.scatter(unatt[(labelling == i), 0],
               unatt[(labelling == i), 1],
               c=c,
               label=label,
               s=5,
               picker=True)

#ax.scatter(output_2d[:, 0], output_2d[:, 1], c=target)
plt.legend()
plt.show()
exit(0)
#HEED MY WORDS AND DO NOT MOVE PAST THIS POINT

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

