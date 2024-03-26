from sklearn.manifold import TSNE
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join

def plot_tsne(X, labels, epoch, args):
    norm = plt.Normalize(0,1)
    labels = labels.numpy()
    colors = np.array(plt.cm.viridis(norm(np.linspace(0,1,len(np.unique(labels))))))
    tsne = TSNE(n_components = 2, learning_rate='auto',init='random')
    X = X.numpy()
    x0 = tsne.fit_transform(X)
    x = x0[:, 0]
    y = x0[:, 1]

    plt.scatter(x,y,c=colors[labels])
    plt.savefig(join(args.tsne_dir, f'{epoch}.png'))