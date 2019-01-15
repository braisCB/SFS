from dataset_reader.nips import load_dataset
from src.utils import balance_data
from keras import callbacks, backend as K
import keras.utils.np_utils as kutils
# from sklearn.svm import SVC
from scripts.ablation.reps.ranker import create_model
import numpy as np
import json
from copy import deepcopy
import os



dataset_names = [
    ('arcene', 7000),
    ('dexter', 9947),
    ('gisette', 2500),
    ('madelon', 20),
    ('dorothea', 50000)
]

reps = 10
limit = 1.0
gamma = 0.0
epochs = 100
b_size = 100

root_directory = './scripts/ablation/reps/info/'
datasets_directory = './datasets/nips/'


def mu(n_features, max_features):
    return 1.5


if __name__ == '__main__':
    os.chdir('../../../')

    for dataset_stats in dataset_names:
        dataset_name, max_features = dataset_stats
        print('loading dataset', dataset_name)
        dataset = load_dataset(dataset_name, directory=datasets_directory, normalize=dataset_name not in ['dexter', 'dorothea'])
        print('data loaded. labels =', dataset['train']['data'].shape)
        batch_size = min(len(dataset['train']['data']), b_size)
        input_shape = dataset['train']['data'].shape[-1:]

        nclasses = len(np.unique(dataset['train']['label']))

        data = dataset['train']['data']
        label = dataset['train']['label']

        data, label = balance_data(data, label)

        valid_data = dataset['validation']['data']
        valid_label = dataset['validation']['label']

        label = kutils.to_categorical(label, 2)
        valid_label = kutils.to_categorical(valid_label, 2)

        total_features = data.shape[-1]

        fs_filename = root_directory + dataset_name + '_gamma_' + str(gamma) + '_ranks.json'
        output_filename = root_directory + dataset_name + '_gamma_' + str(gamma) + '_results.json'
        try:
            with open(fs_filename) as outfile:
                ranks = json.load(outfile)
        except:
            continue

        for method in ranks:
            print('METHOD : ', method)
            rank_list = ranks[method]
            for example in rank_list:
                if 'regularization' in example:
                    print('regularization : ', example['regularization'])
                if 'gamma' in example:
                    print('gamma : ', example['gamma'])
                print('reps : ', example['reps'])
                output = deepcopy(example)
                nfeats = []
                accuracies = []
                rank = np.array(example['rank']).astype(int)
                n_features = 1
                best_acc = 0
                best_feat = 0
                # max_features = int(limit * data.shape[-1])
                while n_features <= max_features:
                    # print('n_features : ', n_features)
                    r_accuracy = []
                    train_accuracy = 0.0
                    data_min = data[:, rank[:n_features]]
                    for rep in range(reps):
                        model = create_model((n_features, ), regularization=example['regularization'])
                        model.fit(data_min, label, epochs=epochs, batch_size=batch_size, verbose=0)
                        train_accuracy += model.evaluate(data_min, label, verbose=0)[-1] * 100
                        r_accuracy.append(model.evaluate(valid_data[:, rank[:n_features]], valid_label, verbose=0)[-1] * 100)
                        del model
                        K.clear_session()
                    accuracy = np.mean(r_accuracy)
                    print('nfeatures : ', n_features, ', acc : ', accuracy)
                    train_accuracy /= reps
                    if accuracy > best_acc:
                        # print("Accuracy : ", accuracy)
                        best_acc = accuracy
                        best_feat = n_features
                    nfeats.append(n_features)
                    accuracies.append(r_accuracy)
                    if n_features == max_features:
                        break
                    n_features = min(int(n_features * mu(n_features, total_features)) + 1, max_features)
                output['classification'] = {
                    'n_features': nfeats,
                    'accuracy': accuracies
                }
                print('best : ', best_acc, ' , feats : ', best_feat)
                accuracies = np.array(np.mean(accuracies, axis=-1))
                nfeats = np.array(nfeats)
                roc = 0.5 * (accuracies[1:] + accuracies[:-1]) * (nfeats[1:] - nfeats[:-1])
                roc = np.sum(roc) / (nfeats[-1] - nfeats[0])
                print('ROC : ', roc)
                output['percentiles'] = []
                for per in [0.1, 0.25, 0.5, 1.0]:
                    n_features = int(max_features * per)
                    r_accuracy = []
                    data_min = data[:, rank[:n_features]]
                    for rep in range(reps):
                        model = create_model((n_features, ), regularization=example['regularization'])
                        model.fit(data_min, label, epochs=epochs, batch_size=batch_size, verbose=0)
                        r_accuracy.append(model.evaluate(valid_data[:, rank[:n_features]], valid_label, verbose=0)[-1] * 100)
                        del model
                        K.clear_session()
                    accuracy = np.mean(r_accuracy)
                    print("Percentile : ", per)
                    print("Accuracy : ", accuracy)
                    print("Error : ", 100 - accuracy)
                    output['percentiles'].append(
                        {
                            'percentile': per,
                            'n_features': n_features,
                            'accuracy': r_accuracy
                        }
                    )

                try:
                    with open(output_filename) as outfile:
                        results = json.load(outfile)
                except:
                    results = {}

                if method not in results:
                    results[method] = []

                results[method].append(output)

                with open(output_filename, 'w') as outfile:
                    json.dump(results, outfile)
