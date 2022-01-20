# Load Python Libraries
import numpy as np
import gzip, pickle
from tsne import bh_sne
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from functions import *
limit = 10

output = np.load('./output.npy').astype(np.float64)[:limit]
data = np.load('./data.npy')
target = np.load('./target.npy')[:limit]
advoutput = np.load('./advoutput.npy').astype(np.float64)[:limit]
advtarget = np.load('./advtarget.npy')[:limit]
advdata = np.load('./advdata.npy')[:limit]
print('data shape: ', data.shape)
print('target shape: ', advtarget.shape)
print('output shape: ', advoutput.shape)

#target = target.tolist()
fig, ax = plt.subplots()

output = np.append(output, advoutput, axis=0)
print('new shape: ', output.shape)

tsne = TSNE(n_components=2, random_state=1)
X_2d = tsne.fit_transform(output)


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

