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


root_directory = './info/'
methods = ['elastic_net', 'dfs', 'lasso', 'mim', 'reliefF']

if __name__ == '__main__':
    for dataset_name, regularization in dataset_names:
        for method in methods:
            print('DATASET : ', dataset_name)
            print('METHOD : ', method)
            fs_filename = root_directory + dataset_name + '_' + method + '_results.json'

            try:
                with open(fs_filename) as outfile:
                    info_data = json.load(outfile)
            except:
                continue

            for name in info_data:

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

                roc = 0.5 * (accuracy[1:] + accuracy[:-1]) * (n_features[1:] - n_features[:-1])
                roc = np.sum(roc) / (n_features[-1] - n_features[0])

                print('dataset: ', dataset_name, ', name : ', name,
                      ', acc : ', best_accuracy, ', feats : ', best_features,
                      ', first : ', accuracy[0])
