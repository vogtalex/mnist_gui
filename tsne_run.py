# Load Python Libraries
import numpy as np
import gzip, pickle
from tsne import bh_sne
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


output = np.load('./output.npy').astype(np.float64)
data = np.load('./data.npy')
target = np.load('./target.npy')
print('data shape: ', data.shape)
print('target shape: ', target.shape)
print('output shape: ', output.shape)

#target = target.tolist()

labels = list(range(0, 10))
target_ids = range(10)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'

tsne = TSNE(n_components=2, random_state=1)
X_2d = tsne.fit_transform(output)

print(X_2d)
print(X_2d[0])
print(X_2d[1, 0])

for i, c, label in zip(target_ids, colors, labels):
    plt.scatter(X_2d[target[...,0] == i, 0], X_2d[target[...,0] == i, 1], c=c, label=label)
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

