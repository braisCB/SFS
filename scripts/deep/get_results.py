import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_names = [
    'fashion_mnist', # 'cifar_10', 'cifar_100', 'fashion_mnist'
]

gamma = 0.9
network_names = ['cnnsimple', 'wrn164']
fs_methods = ['dfs', 'sfs']

if __name__ == '__main__':
    os.chdir('../../')
    for dataset_name in dataset_names:
        directory = './scripts/deep/' + dataset_name + '/info/'

        for network_name in network_names:
            for fs_method in fs_methods:

                fs_filename = directory + network_name + '_' + str(gamma) + '_' + fs_method + '_results.json'

                try:
                    with open(fs_filename) as outfile:
                        info_data = json.load(outfile)
                except:
                    info_data = {}

                for name in info_data:
                    print(fs_method, name)
                    n_features = 0.
                    accuracy = 0.
                    accuracy_2 = 0.
                    nexamples = 0
                    for example in info_data[name]:
                        classification = example['classification']
                        n_features = np.array(classification['n_features'])
                        example_accuracy = np.array(classification['accuracy'])
                        print(example_accuracy)
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


