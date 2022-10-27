import numpy as np
import os
import matplotlib.pyplot as plt
import json

with open('config.json') as f:
    config = json.load(f)

npys = config['Model']['outputDir']
displaySubset = f"subset{config['General']['displaySubset']}"
idx = config["General"]["startIdx"]

# open unattacked and attacked versions of the same image and display them
exdata1 = np.load(os.path.join(npys,'examples',displaySubset,'data.npy')).astype(np.float64)
img = exdata1[idx].reshape((28,28))
plt.figure()
plt.imshow(img, cmap='gray')

exdata2 = np.load(os.path.join(npys,'examples',displaySubset,'advdata.npy')).astype(np.float64)
img = exdata2[idx].reshape((28,28))
plt.figure()
plt.imshow(img, cmap='gray')
plt.show(block=False)

# compute norm between the unattacked/attacked examples
print("L2: ",np.linalg.norm(exdata1[idx] - exdata2[idx]))
print("Linf: ",np.linalg.norm(exdata1[idx] - exdata2[idx], ord=np.Inf))

input("Press Enter to continue...")
