# Load Python Libraries
import numpy as np
import os
import gzip
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from functions import *
import torch


def generateTSNE(idx):
    limit = 10000

    npys = './npys'
    eps = 'e3'
    examples = 'examples'

    # train data
    trainlabels = np.load(os.path.join(
        npys, 'trainlabels.npy')).astype(np.float64)[:limit]
    trainoutput = np.load(os.path.join(
        npys, 'trainoutput.npy')).astype(np.float64)[:limit]
    traindata = np.load(os.path.join(npys, 'traindata.npy')
                        ).astype(np.float64)[:limit]

    # adversarial data
    testlabels = np.load(os.path.join(
        npys, eps, 'testlabels.npy')).astype(np.float64)[:limit]
    advoutput = np.load(os.path.join(
        npys, eps, 'advoutput.npy')).astype(np.float64)[:limit]
    advdata = np.load(os.path.join(npys, eps, 'advdata.npy')
                      ).astype(np.float64)[:limit]

    # example data
    exlabels = np.load(os.path.join(npys, examples, 'testlabels.npy')).astype(
        np.float64)[:limit]
    exoutput = np.load(os.path.join(npys, examples, 'advoutput.npy')).astype(
        np.float64)[:limit]
    exdata = np.load(os.path.join(npys, examples, 'advdata.npy')
                     ).astype(np.float64)[:limit]

    print("advdata ", advdata.shape)
    print("testlabels ", testlabels.shape)
    print("advoutput ", advoutput.shape)

    def findNearest():
        k = 10
        print("Index: ", idx)
        example = exdata[idx]
        label = np.argmax(exoutput[idx])
        print("Model prediction: ", label)

        l = advdata - example

        norms = np.linalg.norm(l, axis=1)

        top = np.argpartition(norms, k-1)

        ########
        print("True label: ", int(exlabels[idx]))
        print("Nearest 10 labels: ")
        print(top[:k])
        print([(int(testlabels[i])) for i in top[:k]])
        #print("Distance to nearest 10 points: ")
        print([(norms[idx]) for idx in top[1:k]])
        return norms, top[1:k], label, int(exlabels[idx])

    norms, idxs, prediction, truelabel = findNearest()
    print(norms)

    print('data shape: ', traindata.shape)
    print('labels shape: ', trainlabels.shape)
    print('output shape: ', trainoutput.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # for combining data/advdata
    #data = np.append(data, advdata, axis=0)

    X_2d = []
    # if exists("./npys/e1/embedding.npy"):
    if exists(os.path.join(npys, eps, 'embedding.npy')):
        #X_2d = np.load('./npys/e1/embedding.npy').astype(np.float64)
        X_2d = np.load(os.path.join(npys, eps, 'embedding.npy')
                       ).astype(np.float64)
    else:
        tsne = TSNE(n_components=2, random_state=3, perplexity=100)
        X_2d = tsne.fit_transform(traindata)
        np.save('./embedding.npy', X_2d, allow_pickle=False)

    labels = list(range(0, 10))
    target_ids = range(10)
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'

    #plt.scatter(X_2d[:limit, 0], X_2d[:limit, 1], s=5, c='r', label='normal')
    #plt.scatter(X_2d[limit:, 0], X_2d[limit:, 1], s=2, c='b', label='attacked')

    # label for class
    '''
    for i, c, label in zip(target_ids, colors, labels):
        ax1.scatter(X_2d[(trainlabels[...] == i), 0],
                X_2d[(trainlabels[...] == i), 1],
                c=c,
                label=label,
                s=3,
                picker=True)
    '''

    for i, c, label in zip(target_ids, colors, labels):
        ax1.scatter(X_2d[(testlabels[...] == i), 0],
                    X_2d[(testlabels[...] == i), 1],
                    c=c,
                    label=label,
                    s=3,
                    picker=True)

    ax1.set_title("Test Data")

    # label for norms
    """
    for i, c, label in zip(target_ids, colors, labels):
        ax2.scatter(X_2d[(trainlabels == i), 0],
                X_2d[(trainlabels == i), 1],
                c=norms[(trainlabels == i)],
                s=3,
                picker=True,
                cmap='viridis')
    """

    # for i, c, label in zip(target_ids, colors, labels):
    ax2.scatter(X_2d[..., 0],
                X_2d[..., 1],
                c=norms[...],
                s=3,
                cmap='viridis')

    # ax.scatter(att[..., 0],
    #           att[..., 1],
    #           c='fuchsia',
    #           label="attacked",
    #           s=3,
    #           picker=True)

    cb = ax2.scatter(X_2d[idxs, 0], X_2d[idxs, 1],
                     c='black',
                     label="nearest",
                     s=10,
                     picker=True)

    print('max distance', max(norms))
    print('min distance', min(norms))
    print('avg distance', sum(norms)/len(norms))

    title = "Model Prediction: %d\nActual Label: %d\nAverage Distance: %f" % (
        prediction, truelabel, float(sum(norms))/len(norms))
    ax2.set_title(title)

    '''
    ax2.scatter(adv[idx,0],att[idx,1],
            c='red',
            label="attacked",
            s=10,
            picker=True)
    '''

    # p=ax2.get_children()[2]
    plt.colorbar(cb, label="norm")
    cb.set_clim(5, 15)

    #ax.scatter(output_2d[:, 0], output_2d[:, 1], c=target)
    ax1.legend()
    plt.tight_layout()
    plt.figure(figsize=(4,3))
    return fig
