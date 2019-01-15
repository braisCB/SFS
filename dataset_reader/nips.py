import os
import numpy as np
from scipy import sparse as sp
from keras.utils.data_utils import get_file


datasets_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


def download_dataset(dataset_name, directory=None):

    if dataset_name.lower() not in ['arcene', 'dexter', 'dorothea', 'madelon', 'gisette']:
        raise Exception(dataset_name + ' is not a known dataset')

    directory = ('./datasets' if directory is None else directory) + '/' + dataset_name.lower() + '/'

    if not os.path.isdir(directory):
        os.makedirs(directory)

    train_url = datasets_url + dataset_name.lower() + '/'

    for subset in ['train', 'test', 'valid']:
        for option in ['labels', 'data']:
            if option == 'labels' and subset == 'test':
                continue
            filename = dataset_name + '_' + subset + '.' + option
            if subset == 'valid' and option == 'labels':
                url = train_url + dataset_name.lower() + '_' + subset + '.' + option
            else:
                url = train_url + dataset_name.upper() + '/' + dataset_name.lower() + '_' + subset + '.' + option
            if not os.path.exists(directory + '/' + filename):
                get_file(filename, url, cache_dir=directory, cache_subdir='.')


def load_dataset(dataset_name, directory=None, normalize=True, sparse=False):
    download_dataset(dataset_name, directory=directory)
    directory = ('./datasets' if directory is None else directory) + '/' + dataset_name.lower() + '/'
    dataset = load_data(directory + '/' + dataset_name, normalize=normalize, sparse=sparse)
    return dataset


def load_data(source, normalize=True, sparse=False):
    info = {
        'train': {}, 'validation': {}, 'test': {}
    }

    file = source + '_train.labels'
    info['train']['label'] = np.loadtxt(file, dtype=np.int16)
    info['train']['label'][info['train']['label'] < 0] = 0

    file = source + '_train.data'
    try:
        info['train']['data'] = np.loadtxt(file, dtype=np.int16).astype(np.float32)
    except:
        if 'dexter' in source:
            info['train']['data'] = __load_dexter_data(file, sparse)
        else:
            info['train']['data'] = __load_dorothea_data(file, sparse)

    file = source + '_test.data'
    try:
        info['test']['data'] = np.loadtxt(file, dtype=np.int16).astype(np.float32)
    except:
        if 'dexter' in source:
            info['test']['data'] = __load_dexter_data(file, sparse)
        else:
            info['test']['data'] = __load_dorothea_data(file, sparse)

    file = source + '_valid.labels'
    info['validation']['label'] = np.loadtxt(file, dtype=np.int16)
    info['validation']['label'][info['validation']['label'] < 0] = 0

    file = source + '_valid.data'
    try:
        info['validation']['data'] = np.loadtxt(file, dtype=np.int16).astype(np.float32)
    except:
        if 'dexter' in source:
            info['validation']['data'] = __load_dexter_data(file, sparse)
        else:
            info['validation']['data'] = __load_dorothea_data(file, sparse)

    if normalize:
        if 'gisette' in source:
            info['train']['data'][info['train']['data'] < 3] = 0.
            info['train']['data'][info['train']['data'] > 0] = 1.
            info['validation']['data'][info['validation']['data'] < 3] = 0.
            info['validation']['data'][info['validation']['data'] > 0] = 1.
            info['test']['data'][info['test']['data'] < 3] = 0.
            info['test']['data'][info['test']['data'] > 0] = 1.
        else:
            axis = None # 0 if 'madelon' in source else None
            mean = np.mean(info['train']['data'], axis=axis)
            std = np.std(info['train']['data'], axis=axis)
            info['train']['data'] = (info['train']['data'] - mean) / np.maximum(1e-6, std)
            info['validation']['data'] = (info['validation']['data'] - mean) / np.maximum(1e-6, std)
            info['test']['data'] = (info['test']['data'] - mean) / np.maximum(1e-6, std)
            # train_min = np.min(info['train']['data'], axis=0, keepdims=True)
            # train_max = np.max(info['train']['data'], axis=0, keepdims=True)
            # info['train']['data'] = (info['train']['data'] - train_min) / np.maximum(1e-8, train_max - train_min)
            # info['validation']['data'] = (info['validation']['data'] - train_min) / np.maximum(1e-8, train_max - train_min)
            # info['test']['data'] = (info['test']['data'] - train_min) / np.maximum(1e-8, train_max - train_min)

    return info


def __load_dexter_data(source, sparse):
    """
    A function that reads in the original dexter data in sparse form of feature:value
    and transform them into matrix form.
    # Arguments:
    filename: the url to either the dexter_train.data or dexter_valid.data
    mode: either 'text' for unpacked file; 'gz' for .gz file; or 'online' to download from the UCI repo
    # Return:
    the dexter data in matrix form.
    """
    with open(source) as f:
        readin_list = f.readlines()

    def to_dense_sparse(string_array):
        n = len(string_array)
        inds = np.zeros(n, dtype='int32')
        vals = np.zeros(n, dtype='int32')
        ret = np.zeros(20000, dtype='int32')
        for i in range(n):
            this_split = string_array[i].split(':')
            inds[i] = int(this_split[0]) - 1
            vals[i] = int(this_split[1])
        ret[inds] = vals
        return ret

    N = len(readin_list)
    dat = [None]*N

    for i in range(N):
        dat[i] = to_dense_sparse(readin_list[i].split(' ')[0:-1])[None, :]

    dat = np.concatenate(dat, axis=0).astype('float32')
    # PREPROCESSING
    dat[dat < 3] = 0
    dat[dat > 0] = 1

    if sparse:
        dat = sp.csr_matrix(dat)
    return dat


def __load_dorothea_data(filename, sparse):
    """
    A function that reads in the original dorothea data in sparse form of feature:value
    and transform them into matrix form.
    # Arguments:
    filename: the url to either the dorothea_train.data or dorothea_valid.data
    mode: either 'text' for unpacked file; 'gz' for .gz file; or 'online' to download from the UCI repo
    # Return:
    the dexter data in matrix form.
    """
    with open(filename) as f:
        readin_list = f.readlines()

    def to_dense_dorothea(string_array):
        n = len(string_array)
        inds = np.zeros(n, dtype='int32')
        ret = np.zeros(100001, dtype='int32')
        for i in range(n):
            this_split = string_array[i].split(' ')
            inds[i] = int(this_split[0])
        ret[inds] = 1
        return ret

    N = len(readin_list)
    dat = [None]*N

    for i in range(N):
        dat[i] = to_dense_dorothea(readin_list[i].split(' ')[1:-1])[None, :]

    dat = np.concatenate(dat, axis=0).astype('float32')[:, 1::]
    if sparse:
        dat = sp.csr_matrix(dat)
    return dat  # the first column all zero

