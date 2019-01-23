from src.SFS import get_rank
from keras import callbacks
from dataset_reader.regression import load_dataset
from scripts.regression import network_models
import json
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import KFold
import numpy as np


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))


gamma = 0.
b_size = 128
epochs = 600
reps = 2

dataset_names = ['energy-molecule']
sd_directory = './scripts/regression/energy_molecule/info/'
dataset_directory = './datasets/regression/'
network_name = 'dense'
methods = ['sfs', 'dfs']

np.random.seed(42)


def scheduler(epoch, wrn=False):
    if epoch < 150:
        return 0.01
    elif epoch < 300:
        return 0.002
    elif epoch < 450:
        return 0.0004
    else:
        return 0.00008


rank_kwargs = {
    'reps': reps,
    'gamma': gamma,
    'epsilon': 60
}

kfold = KFold(n_splits=5, shuffle=True)


def main():

    for dataset_name in dataset_names:
        print('dataset =', dataset_name)
        dataset = load_dataset(dataset_name, directory=dataset_directory)

        ids = np.unique(dataset['id'])
        print('NUMBER OF IDS : ', ids)
        data = dataset['data']
        result = dataset['result']

        for train_index, test_index in kfold.split(data):

            train_data, test_data = data[train_index], data[test_index]
            train_result, test_result = result[train_index], result[test_index]

            model_func = getattr(network_models, network_name.split('_')[0])

            batch_size = min(len(train_data), b_size)

            fit_kwargs = {
                'epochs': epochs,
                'callbacks': [
                    callbacks.LearningRateScheduler(scheduler)
                ],
                'verbose': 2
            }

            generator = dataset['generator']
            generator_kwargs = {
                'batch_size': batch_size
            }
            fit_kwargs['batch_size'] = batch_size

            for fs_mode in methods:
                for lasso in [0., 5e-4]:
                    if lasso == 0.0 and fs_mode == 'dfs':
                        continue
                    print('reps : ', reps)
                    print('method : ', fs_mode)
                    for regularization in [5e-4]:
                        name = dataset_name + '_' + fs_mode + '_l_' + str(lasso) + '_g_' + str(gamma) + \
                               '_r_' + str(regularization)
                        print(name)
                        model_kwargs = {
                            'lasso': lasso,
                            'regularization': regularization
                        }
                        rank = get_rank(fs_mode, data=train_data, label=train_result, model_func=model_func,
                                        model_kwargs=model_kwargs, fit_kwargs=fit_kwargs, generator=generator,
                                        generator_kwargs=generator_kwargs, rank_kwargs=rank_kwargs, type='regression')
                        try:
                            os.makedirs(sd_directory)
                        except:
                            pass
                        output_filename = sd_directory + dataset_name + '_' + str(gamma) + '_rank.json'

                        try:
                            with open(output_filename) as outfile:
                                info_data = json.load(outfile)
                        except:
                            info_data = {}

                        key = fs_mode + '_' + str(lasso)
                        if key not in info_data:
                            info_data[key] = []

                        info_data[key].append(
                            {
                                'lasso': lasso,
                                'gamma': gamma,
                                'regularization': regularization,
                                'rank': rank.tolist(),
                                'reps': reps,
                                'train_index': train_index.tolist(),
                                'test_index': test_index.tolist()
                            }
                        )

                        with open(output_filename, 'w') as outfile:
                            json.dump(info_data, outfile)

                        del rank


if __name__ == '__main__':
    os.chdir('../../../')
    main()