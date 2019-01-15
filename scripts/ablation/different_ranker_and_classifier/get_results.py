import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_names = [
    ('arcene', 1e-1),
    ('dexter', 1e-1),
    ('madelon', 1e-1),
    ('dorothea', 1e-1),
    ('gisette', 1e-1),
]

gamma = 0.975

root_directory = './scripts/ablation/different_ranker_and_classifier/info_p/'
order = ['linear', 'poly', 'rbf', 'sigmoid']


if __name__ == '__main__':
    os.chdir('../../../')
    for dataset_name, regularization in dataset_names:

        fs_filename = root_directory + dataset_name + '_gamma_' + str(gamma) + '_ranks.json'

        try:
            with open(fs_filename) as outfile:
                info_data = json.load(outfile)
        except:
            info_data = {}

        for c in order:
            for k in order:
                name = 'sfs_k_' + k + '_c_' + c

                if name not in info_data:
                    continue

                print(name)
                n_features = 0.
                accuracy = 0.
                nexamples = 0
                for example in info_data[name]:
                    classification = example['classification']
                    n_features = np.array(classification['n_features'])
                    example_accuracy = np.array(classification['accuracy'])
                    accuracy += np.sum(example_accuracy, axis=-1)
                    nexamples += example_accuracy.shape[-1]
                accuracy /= nexamples

                best_accuracy = np.max(accuracy)
                best_index = np.where(accuracy == best_accuracy)[0]
                best_features = np.min(n_features[best_index])
                print('dataset: ', dataset_name, ', name : ', name, ', acc : ', best_accuracy, ', feats : ', best_features)
