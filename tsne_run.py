# Load Python Libraries
import numpy as np
import gzip, pickle
from tsne import bh_sne

output = np.load('./output.npy').astype(np.float64)
data = np.load('./data.npy')
target = np.load('./target.npy')
print('data shape: ', data.shape)
print('target shape: ', target.shape)
print('output shape: ', output.shape)

output_2d = bh_sne(output)

np.save('train/output_2d.npy', output_2d, allow_pickle=False)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(output_2d[:, 0], output_2d[:, 1], c=target)
plt.savefig('train/output_2d.png', bbox_inches='tight')
plt.show()

