# Load Python Libraries
import numpy as np
import os
import gzip, pickle
#import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from functions import *
import torch
def generateTSNEPlots(idx, plotID):
    limit = 10000

    npys = './cifar_npys'
    eps = 'e1'
    exeps = 'e1'
    examples = 'examples'
    tests = 'test'
    trains = 'train'

    def get_data(npys,eps,examples):
        #train data
        trainlabels = np.load(os.path.join(npys,trains,'trainlabels.npy')).astype(np.float64)[:limit]
        trainoutput = np.load(os.path.join(npys,trains,'trainoutput.npy')).astype(np.float64)[:limit]
        traindata = np.load(os.path.join(npys,trains,'traindata.npy')).astype(np.float64)[:limit]

        #adversarial data
        testlabels = np.load(os.path.join(npys,tests,eps,'testlabels.npy')).astype(np.float64)[:limit]
        advoutput = np.load(os.path.join(npys,tests,eps,'advoutput.npy')).astype(np.float64)[:limit]
        advdata = np.load(os.path.join(npys,tests,eps,'advdata.npy')).astype(np.float64)[:limit]

        #example data
        exlabels = np.load(os.path.join(npys,examples,exeps,'testlabels.npy')).astype(np.float64)[:limit]
        exoutput = np.load(os.path.join(npys,examples,exeps,'advoutput.npy')).astype(np.float64)[:limit]
        exdata = np.load(os.path.join(npys,examples,exeps,'advdata.npy')).astype(np.float64)[:limit]
        return trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata

    trainlabels, trainoutput, traindata, testlabels, advoutput, advdata, exlabels, exoutput, exdata = get_data(npys,eps,examples)

    # print("advdata ",advdata.shape)
    # print("testlabels ",testlabels.shape)
    # print("advoutput ",advoutput.shape)

    def findNearest():
        k=10
        # print("Index: ",idx)
        example = exdata[idx]
        label = np.argmax(exoutput[idx])
        # print("Model prediction: ", label)

        l = advdata - example

        norms = np.linalg.norm(l,axis=1)


        top = np.argpartition(norms,k-1)

        # print("True label: ", int(exlabels[idx]))
        # print("Nearest 10 labels: ")
        # print(top[:k])
        # print([(int(testlabels[i])) for i in top[:k]])
        # #print("Distance to nearest 10 points: ")
        # print([(norms[idx]) for idx in top[1:k]])
        return norms, top[1:k],label,int(exlabels[idx])

    norms,idxs,prediction,truelabel = findNearest()
    # print(norms)

    print('data shape: ', traindata.shape)
    print('labels shape: ', trainlabels.shape)
    print('output shape: ', trainoutput.shape)



    #for combining data/advdata
    #data = np.append(data, advdata, axis=0)

    X_2d = []
    if exists(os.path.join(npys,tests,eps,'embedding.npy')):
        X_2d = np.load(os.path.join(npys,tests,eps,'embedding.npy')).astype(np.float64)
    else:
        tsne = TSNE(n_components=2, random_state=3,perplexity=100,verbose=1)
        X_2d = tsne.fit_transform(traindata)
        np.save('./embedding.npy', X_2d, allow_pickle=False)

    #img = exdata[idx].reshape((28,28))

    ###HISTOGRAMS###################
generateTSNEPlots(10,0)

'''
    if plotID == 1:
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

        return fig1

    if plotID == 0:

        labels = list(range(0, 10))
        target_ids = range(10)
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'orange', 'purple'

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
        return fig
        '''
