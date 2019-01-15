from dataset_reader.nips import load_dataset
from src.SFS import get_rank
from src.sklearn_parser import SVC
from sklearn.svm import SVC as sklearn_SVC
import keras.utils.np_utils as kutils
import numpy as np
import json
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_names = [
    ('arcene', 1e-1),
    # ('dexter', 1e-1),
    # ('madelon', 1e-3),
    # ('dorothea', 1e-1),
    # ('gisette', 1e-3),
]

gamma = 0.
classifier_gamma = .975
b_size = 100
epochs = 200
reps = 1
lasso = 0.0

Cs = [.1, 1.5, 10., 100.]

np.random.seed(1)

root_directory = './scripts/ablation/overfitting/info/'
datasets_directory = './datasets/nips/'

if __name__ == '__main__':
    os.chdir('../../../')
    for dataset_name, regularization in dataset_names:

        fs_filename = root_directory + dataset_name + '_gamma_' + str(gamma) + '_ranks.json'

        print('loading dataset', dataset_name)
        dataset = load_dataset(
            dataset_name, directory=datasets_directory, normalize=dataset_name not in ['dexter', 'dorothea'],
            sparse=True
        )
        print('data loaded. labels =', dataset['train']['data'].shape)
        input_shape = dataset['train']['data'].shape[-1:]
        batch_size = min(dataset['train']['data'].shape[0], b_size)

        uclasses = np.unique(dataset['train']['label'])
        nclasses = len(uclasses)

        data = dataset['train']['data']
        label = dataset['train']['label']
        label = kutils.to_categorical(label, nclasses)
        valid_data = dataset['validation']['data']
        valid_label = kutils.to_categorical(dataset['validation']['label'], nclasses)

        label_argmax = np.argmax(label, axis=-1)
        valid_label_argmax = np.argmax(valid_label, axis=-1)

        for c in Cs:
            for kernel in ['rbf']:
                print('C : ', c, ' , kernel : ', kernel)
                name = 'sfs_k_' + kernel + '_c_' + str(c)
                print(name)
                model_kwargs = {
                    'C': c,
                    'degree': 3.,
                    'coef0': 1. * (kernel == 'poly'),
                    'kernel': kernel,
                    'class_weight': 'balanced',
                    'cache_size': 4096,
                    'max_iter': 10000
                }
                fit_kwargs = {
                }
                evaluate_kwargs = {
                    'verbose': 0,
                    'batch_size': batch_size
                }
                rank_kwargs = {
                    'gamma': gamma,
                    'reps': reps
                }
                saliency_kwargs = {
                    'batch_size': 16,
                }
                result = get_rank('sfs', data=data, label=label, model_func=SVC,
                                  rank_kwargs=rank_kwargs, fit_kwargs=fit_kwargs, model_kwargs=model_kwargs,
                                  saliency_kwargs=saliency_kwargs,
                                  return_info=True, valid_data=valid_data, valid_label=valid_label)

                rank = result['rank']

                model_kwargs['C'] = 1.5
                n_features = data.shape[-1]
                nfeats = []
                accuracies = []
                train_accuracies = []
                while n_features:
                    n_accuracies = []
                    train_n_accuracies = []
                    for _ in range(reps):
                        model = sklearn_SVC(**model_kwargs)
                        model.fit(data[:, rank[:n_features]], label_argmax, **fit_kwargs)
                        n_accuracies.append(model.score(valid_data[:, rank[:n_features]], valid_label_argmax))
                        train_n_accuracies.append(model.score(data[:, rank[:n_features]], label_argmax))
                        print(
                            'n_features : ', n_features, ', acc : ', n_accuracies[-1], ', train_acc : ', train_n_accuracies[-1]
                        )
                        del model
                    accuracies.append(n_accuracies)
                    train_accuracies.append(train_n_accuracies)
                    nfeats.append(n_features)
                    n_features = int(n_features * classifier_gamma)

                try:
                    with open(fs_filename) as outfile:
                        info_data = json.load(outfile)
                except:
                    info_data = {}

                if name not in info_data:
                    info_data[name] = []

                info_data[name].append(
                    {
                        'lasso': lasso,
                        'gamma': gamma,
                        'regularization': regularization,
                        'rank': result['rank'],
                        'classification': {
                            'n_features': nfeats,
                            'accuracy': accuracies,
                            'train_accuracy': train_n_accuracies
                        },
                        'reps': reps,
                        'model_kwargs': model_kwargs,
                        'C': c
                    }
                )

                if not os.path.isdir(root_directory):
                    os.makedirs(root_directory)

                with open(fs_filename, 'w') as outfile:
                    json.dump(info_data, outfile)

                del result
