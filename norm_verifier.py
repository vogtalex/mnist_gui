import numpy as np
import os
import matplotlib.pyplot as plt
import json

limit = 9000

with open('config.json') as f:
    config = json.load(f)

npys = config['Model']['outputDir']
#what attack level of example point
exeps = f"e{config['General']['displayEpsilon']}"
examples = 'examples'

idx = 32

exdata1 = np.load(os.path.join(npys,examples,'e0','advdata.npy')).astype(np.float64)[:limit]
img = exdata1[idx].reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

exdata2 = np.load(os.path.join(npys,examples,exeps,'advdata.npy')).astype(np.float64)[:limit]
img = exdata2[idx].reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

print("L2: ",np.linalg.norm(exdata1[idx] - exdata2[idx]))
print("Linf: ",np.linalg.norm(exdata1[idx] - exdata2[idx],ord=np.Inf))
