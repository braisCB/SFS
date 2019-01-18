import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_names = [
    'slice_localization_data'
]

gamma = 0.9

if __name__ == '__main__':
    os.chdir('../../')
    for dataset_name in dataset_names:
        directory = './scripts/regression/info/'

        fs_filename = directory + dataset_name + '_' + str(gamma) + '_result.json'

        try:
            with open(fs_filename) as outfile:
                info_data = json.load(outfile)
        except:
            info_data = {}

        for name in info_data:
            print(name)
            n_features = 0.
            accuracy = 0.
            accuracy_2 = 0.
            nexamples = 0
            for classification in info_data[name]:
                n_features = np.array(classification['n_features'])
                example_accuracy = np.array(classification['maes'])
                accuracy += np.sum(example_accuracy, axis=-1)
                accuracy_2 += np.sum(example_accuracy * example_accuracy, axis=-1)
                nexamples += example_accuracy.shape[-1]
            accuracy /= nexamples
            accuracy_2 = np.sqrt(accuracy_2 / nexamples - accuracy * accuracy)

            print(n_features)
            print(accuracy)
            print(accuracy_2)

            best_accuracy = np.max(accuracy)
            best_index = np.where(accuracy == best_accuracy)[0]
            best_features = np.min(n_features[best_index])

            roc = 0.5 * (accuracy[1:] + accuracy[:-1]) * (n_features[1:] - n_features[:-1])
            roc = np.sum(roc) / (n_features[-1] - n_features[0])


