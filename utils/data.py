import pickle
import sys

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp


def load_data(prefix='DBLP'):
    from data_loader import data_loader
    dl = data_loader('../../data/'+prefix)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    labels = np.load('../../data/'+prefix+'/label.npy', allow_pickle=True)

    train = np.load('../../data/' + prefix + "/train.npy", allow_pickle=True)
    test = np.load('../../data/'+prefix + "/test.npy", allow_pickle=True)
    val = np.load('../../data/'+prefix + "/val.npy", allow_pickle=True)

    train_val_test_idx = {}
    train_val_test_idx['train_10'] = train[0]
    train_val_test_idx['train_20'] = train[1]
    train_val_test_idx['train_30'] = train[2]
    train_val_test_idx['val_10'] = val[0]
    train_val_test_idx['val_20'] = val[1]
    train_val_test_idx['val_30'] = val[2]
    train_val_test_idx['test_10'] = test[0]
    train_val_test_idx['test_20'] = test[1]
    train_val_test_idx['test_30'] = test[2]
    return features,\
        adjM, \
        labels,\
        train_val_test_idx,\
        dl
