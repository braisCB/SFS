from dataset_reader.nips import load_dataset
from src.utils import balance_data
import numpy as np
import json
import os
from sklearn.linear_model import ElasticNetCV


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_names = [
    ('madelon', 1e-1),
    ('dexter', 0.1),
    ('arcene', 0.1),
    ('dorothea', 0.1),
    ('gisette', 1e-1),
]

b_size = 100
epochs = 100
gamma = 0.975
lasso = 0.0

root_directory = './scripts/nips/info/'
datasets_directory = './datasets/nips/'


if __name__ == '__main__':
    os.chdir('../../')

    for dataset_name, regularization in dataset_names:

        fs_filename = root_directory + dataset_name + '_elastic_net_results.json'

        print('loading dataset', dataset_name)
        dataset = load_dataset(dataset_name, directory=datasets_directory, normalize=dataset_name not in ['dexter', 'dorothea'])
        print('data loaded. labels =', dataset['train']['data'].shape)
        input_shape = dataset['train']['data'].shape[-1:]
        batch_size = min(len(dataset['train']['data']), b_size)

        uclasses = np.unique(dataset['train']['label'])
        nclasses = len(uclasses)

        data = dataset['train']['data']
        label = dataset['train']['label']
        data, label = balance_data(data, label)
        print('data size after balance : ', data.shape)
        valid_data = dataset['validation']['data']
        valid_label = dataset['validation']['label']

        model_kwargs = {
            'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
            'cv': 5,
            'random_state': 42
        }

        name = 'lasso'
        reg = ElasticNetCV(**model_kwargs).fit(data, label)

        pred = reg.predict(valid_data)
        pred = np.round(pred)
        pred = np.clip(pred, 0, 1).astype(int)
        accuracy = (pred == valid_label).mean()
        print('dataset : ', dataset_name, ' , acc : ', accuracy)

        try:
            with open(fs_filename) as outfile:
                info_data = json.load(outfile)
        except:
            info_data = {}

        if name not in info_data:
            info_data[name] = []

        info_data[name].append(
            {
                'lasso': reg.alpha_,
                'accuracy': accuracy,
                'model_kwargs': model_kwargs
            }
        )

        if not os.path.isdir(root_directory):
            os.makedirs(root_directory)

        with open(fs_filename, 'w') as outfile:
            json.dump(info_data, outfile)

        del data, label, valid_data, valid_label, dataset, info_data