from dataset_reader.nips import load_dataset
from src.SFS import get_rank
from src.sklearn_parser import SVC
from sklearn.svm import SVC as sklearn_SVC
import keras.utils.np_utils as kutils
import numpy as np
import json
import os


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    dataset_names = [
        ('arcene', 1e-1),
        ('dexter', 1e-1),
        ('madelon', 1e-1),
        ('dorothea', 1e-1),
        ('gisette', 1e-1),
    ]

    gamma = 0.975
    b_size = 100
    reps = 1
    lasso = 0.0
    fs_mode = 'sfs'

    np.random.seed(1)

    root_directory = './scripts/ablation/different_ranker_and_classifier/info/'
    datasets_directory = './datasets/nips/'

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

        for kernel in ['sigmoid', 'linear', 'poly', 'rbf']:
            print('kernel : ', kernel)
            name = fs_mode + '_k_' + kernel + '_c_' + kernel
            print(name)
            model_kwargs = {
                'C': 1.,
                'degree': 3.,
                'coef0': 1. * (kernel == 'poly'),
                'kernel': kernel,
                'class_weight': 'balanced',
                'cache_size': 4096,
                'max_iter': 5000
            }
            fit_kwargs = {
            }
            rank_kwargs = {
                'gamma': gamma,
                'reps': reps
            }
            saliency_kwargs = {
                'batch_size': 16,
            }
            result = get_rank(fs_mode, data=data, label=label, model_func=SVC,
                              rank_kwargs=rank_kwargs, fit_kwargs=fit_kwargs, model_kwargs=model_kwargs,
                              saliency_kwargs=saliency_kwargs,
                              return_info=True, valid_data=valid_data, valid_label=valid_label)

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
                    'classification': result['classification'],
                    'reps': reps,
                    'model_kwargs': model_kwargs
                }
            )

            if not os.path.isdir(root_directory):
                os.makedirs(root_directory)

            with open(fs_filename, 'w') as outfile:
                json.dump(info_data, outfile)

            rank = result['rank']

            for kernel_classifier in ['linear', 'poly', 'sigmoid', 'rbf']:
                if kernel == kernel_classifier:
                    continue
                model_kwargs['kernel'] = kernel_classifier
                model_kwargs['coef0'] = 1. * (kernel_classifier == 'poly')
                n_features = data.shape[-1]
                classifier_name = fs_mode + '_k_' + kernel + '_c_' + kernel_classifier
                print(classifier_name)
                nfeats = []
                accuracies = []
                while n_features:
                    n_accuracies = []
                    for _ in range(reps):
                        model = sklearn_SVC(**model_kwargs)
                        model.fit(data[:, rank[:n_features]], label_argmax, **fit_kwargs)
                        n_accuracies.append(model.score(valid_data[:, rank[:n_features]], valid_label_argmax))
                        print(
                            'n_features : ', n_features, ', acc : ', n_accuracies[-1]
                        )
                        del model
                    accuracies.append(n_accuracies)
                    nfeats.append(n_features)
                    n_features = int(n_features * gamma)

                try:
                    with open(fs_filename) as outfile:
                        info_data = json.load(outfile)
                except:
                    info_data = {}

                if classifier_name not in info_data:
                    info_data[classifier_name] = []

                info_data[classifier_name].append(
                    {
                        'lasso': lasso,
                        'gamma': gamma,
                        'regularization': regularization,
                        'rank': result['rank'],
                        'classification': {
                            'n_features': nfeats,
                            'accuracy': accuracies
                        },
                        'reps': reps,
                        'model_kwargs': model_kwargs
                    }
                )

                with open(fs_filename, 'w') as outfile:
                    json.dump(info_data, outfile)

            del result


if __name__ == '__main__':
    os.chdir('../../../')
    main()
