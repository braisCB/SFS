from dataset_reader.nips import load_dataset
import json
import os
from skrebate import ReliefF
from sklearn.svm import SVC as sklearn_SVC


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_names = [
    ('madelon', 1e-3),
    ('arcene', 0.1),
    ('dexter', 0.1),
    ('dorothea', 0.1),
    ('gisette', 1e-1),
]

reps = 1
gamma = 0.975
overwrite = True

root_directory = './scripts/nips/info/'
datasets_directory = './datasets/nips/'


if __name__ == '__main__':
    os.chdir('../../')

    for dataset_name, _ in dataset_names:

        fs_filename = root_directory + dataset_name + '_reliefF_results.json'

        print('loading dataset', dataset_name)
        dataset = load_dataset(dataset_name, directory=datasets_directory, normalize=dataset_name not in ['dexter', 'dorothea'])
        print('data loaded. labels =', dataset['train']['data'].shape)
        input_shape = dataset['train']['data'].shape[-1:]

        data = dataset['train']['data']
        label = dataset['train']['label']
        valid_data = dataset['validation']['data']
        valid_label = dataset['validation']['label']

        mim_kwargs = {
            'random_state': 42,
            'discrete_features': dataset_name in ['dexter', 'dorothea'],
            'n_neighbors': 3
        }

        try:
            with open(fs_filename) as outfile:
                info_data = json.load(outfile)
            for key in info_data:
                rank = info_data[key][0]['rank']
                print('ReliefF rank recovered from ' + fs_filename)
                break
        except:
            info_data = {}
            print('ReliefF started')
            fs = ReliefF(verbose=True, n_jobs=-1)
            fs.fit(data, label)
            rank = fs.top_features_.tolist()
            print('ReliefF finished')

        for kernel in ['linear', 'poly', 'sigmoid', 'rbf']:
            model_kwargs = {
                'C': 1.,
                'degree': 3.,
                'coef0': 1. * (kernel == 'poly'),
                'kernel': kernel,
                'class_weight': 'balanced',
                'cache_size': 4096,
                'max_iter': 10000
            }
            n_features = data.shape[-1]
            name = 'mim_' + kernel
            print(name)
            nfeats = []
            accuracies = []
            while n_features:
                n_accuracies = []
                for _ in range(reps):
                    model = sklearn_SVC(**model_kwargs)
                    model.fit(data[:, rank[:n_features]], label)
                    n_accuracies.append(model.score(valid_data[:, rank[:n_features]], valid_label))
                    del model
                print(
                    'n_features : ', n_features, ', acc : ', n_accuracies
                )
                accuracies.append(n_accuracies)
                nfeats.append(n_features)
                n_features = int(n_features * gamma)

            try:
                with open(fs_filename) as outfile:
                    info_data = json.load(outfile)
            except:
                info_data = {}

            if name not in info_data or overwrite:
                info_data[name] = []

            info_data[name].append(
                {
                    'gamma': gamma,
                    'rank': rank,
                    'classification': {
                        'n_features': nfeats,
                        'accuracy': accuracies
                    },
                    'reps': reps,
                    'model_kwargs': model_kwargs
                }
            )

            if not os.path.isdir(root_directory):
                os.makedirs(root_directory)

            with open(fs_filename, 'w') as outfile:
                json.dump(info_data, outfile)

            del nfeats, accuracies, info_data
        del data, label, valid_label, valid_data, dataset