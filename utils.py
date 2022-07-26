'''
# Time   : 2021/10/25 14:40
# Author : adamwei
# File   : utils.py
'''
import pandas as pd
import numpy as np
import random
import shutil
import struct
import lmdb
import torch.utils.data

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer



def clipping(forward_out, clipthr):
    
    forward_out_norm = np.linalg.norm(forward_out, ord=2, axis=1, keepdims=True)
    
    output = forward_out
    for i in range(len(forward_out_norm)):
        if forward_out_norm[i] > clipthr:
            output[i].__mul__(clipthr/forward_out_norm[i])
        
    return output

def batch_split(data_size, batch_size, batch_type):
    
    original_idxs = range(data_size) 
    
    batch_idxs_list = []
    if batch_size <= 0:
        batch_idxs_list = [i for i in range(data_size)]
    else:
        if batch_type == 'mini-batch':
            num_batchs = int(np.ceil(data_size/batch_size))
            
            
            for i in range(num_batchs):
                if len(original_idxs) > batch_size:
                    batch_idxs = random.sample(original_idxs, batch_size)
                    original_idxs = list(set(original_idxs)-set(batch_idxs))
                else:
                    batch_idxs = original_idxs
                batch_idxs_list.append(batch_idxs)
        else:
            
            batch_idxs_list.append(random.sample(original_idxs, batch_size))
        
    return batch_idxs_list

def majority_label(y):
    N = y.shape[0]

    labels = np.unique(y)
    n_class = labels.shape[0]

    label_cnt = np.zeros(n_class)
    cnt_sum = 0

    for i in range(n_class-1):
        label_cnt[i] = np.count_nonzero(y == labels[i])
        cnt_sum += label_cnt[i]

    label_cnt[n_class-1] = N - cnt_sum
    label_cnt /= float(N)

    return max(label_cnt)


def load_dat(filepath, minmax=None, normalize=False, bias_term=True):
    """ load a dat file

    args:
    minmax: tuple(min, max), dersired range of transformed data
    normalize: boolean, normalize samples individually to unit norm if True
    bias_term: boolean, add a dummy column of 1s
    """
    lines = np.loadtxt(filepath)
    labels = lines[:, -1]
    features = lines[:, :-1]

    N, dim = features.shape
    
    print('Original data shape:', features.shape)

    if minmax is not None:
        minmax = MinMaxScaler(feature_range=minmax, copy=False)
        minmax.fit_transform(features)

    if normalize:
        # make sure each entry's L2 norm is 1
        normalizer = Normalizer(copy=False)
        normalizer.fit_transform(features)

    if bias_term:
        X = np.hstack([np.ones(shape=(N, 1)), features])
    else:
        X = features

    return X, labels

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

def create_avazu_dataset(file_path, data_size=10000):
    
    data = pd.read_csv(file_path, compression='gzip', nrows=data_size)
    
    X = data.fillna('-1')
    
    return X

def create_criteo_dataset(file_path, data_size=10000, embed_dim=8, test_size=0.2):
    data = pd.read_csv(file_path, nrows=data_size)

    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    #Fill in the vacancy
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')

    #Normalization
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    
    #Label Encoding
    for col in sparse_features:
        data[col] = LabelEncoder().fit_transform(data[col])

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
           [[sparseFeature(feat, data[feat].nunique(), embed_dim) for feat in sparse_features]]

    #Dataset split
    X = data.drop(['label'], axis=1).values
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return feature_columns, (X_train, y_train), (X_test, y_test)
