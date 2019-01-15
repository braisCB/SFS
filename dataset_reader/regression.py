import os
import zipfile
import numpy as np
import pandas as pd
from keras.utils.data_utils import get_file


def download_dataset(dataset_name, directory=None):

    directory = './datasets' if directory is None else directory

    if not os.path.isdir(directory):
        os.makedirs(directory)

    if dataset_name == 'slice_localization_data':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip'
        filename = dataset_name + '.zip'
    else:
        raise Exception(dataset_name + ' is not a known dataset')

    if not os.path.exists(directory + '/' + filename):
        get_file(filename, url, cache_dir=directory, cache_subdir='.')

    return directory + '/' + filename


def load_dataset(dataset_name, directory=None, normalize=True):
    directory = './datasets' if directory is None else directory
    filename = download_dataset(dataset_name, directory=directory)
    if dataset_name == 'slice_localization_data':
        archive = zipfile.ZipFile(filename, 'r')
        csv_file = archive.filelist[0].filename
        with archive.open(csv_file) as f:
            data = pd.read_csv(f).values
            id = data[:, 0].astype(int)
            data = data[:, 1:]
            if normalize:
                mean = data.mean(axis=0)
                std = np.maximum(1e-6, data.std(axis=0))
                data = (data - mean) / std
            else:
                mean = 0.0
                std = 1.0
            dataset = {
                'id': id,
                'data': data[:, :-1],
                'result': data[:, -1][:, None],
                'mean': mean,
                'std': std,
                'generator': None
            }
    else:
        raise Exception(dataset_name + ' is not a known dataset')
    return dataset


if __name__ == '__main__':
    dataset = load_dataset('slice_localization_data')

