from dataset_reader.nips import load_dataset
from src.SFS import get_rank
from src.layers import Mask
from src import saliency_function
from src.utils import balance_data
from keras.layers import Activation, Dense, Input, Dropout, BatchNormalization
from sklearn.svm import SVC as sklearn_SVC
from keras.models import Model
from keras import optimizers, callbacks
from keras.regularizers import l2, l1
import keras.utils.np_utils as kutils
import keras.backend as K
import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_names = [
    ('madelon', 0.1),
    ('dexter', 0.1),
    ('arcene', 0.1),
    ('dorothea', 0.1),
    ('gisette', 1e-1),
]

reps = 1
b_size = 100
epochs = 100
gamma = 0.0
gamma_classifier = .975
lasso = 0.001

root_directory = './scripts/nips/info/'
datasets_directory = './datasets/nips/'

kernels_classifier = ['rbf', 'linear', 'poly', 'sigmoid']


def create_model(input_shape, nclasses=2, layer_dims=None, bn=True, kernel_initializer='he_normal',
                 dropout=0.0, lasso=0.0, regularization=0.0, loss_weights=None):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input = Input(shape=input_shape)
    if layer_dims is None:
        layer_dims = [100, 50]
    x = Mask(kernel_regularizer=l1(lasso))(input) if lasso > 0 else input

    for layer_dim in layer_dims:
        x = Dense(layer_dim, use_bias=not bn, kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2(regularization) if regularization > 0.0 else None)(x)
        if bn:
            x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        x = Activation('relu')(x)

    x = Dense(nclasses, use_bias=True, kernel_initializer=kernel_initializer,
              kernel_regularizer=l2(regularization) if regularization > 0.0 else None)(x)
    output = Activation('softmax')(x)

    model = Model(input, output)

    optimizer = optimizers.adam(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    model.saliency = saliency_function.get_saliency('categorical_crossentropy', model)

    return model


def get_scheduler(epochs):
    def scheduler(epoch):
        return .001
    return scheduler


if __name__ == '__main__':
    os.chdir('../../')

    for dataset_name, regularization in dataset_names:

        fs_filename = root_directory + dataset_name + '_dfs_results.json'

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
        label = kutils.to_categorical(label, nclasses)
        valid_data = dataset['validation']['data']
        valid_label = kutils.to_categorical(dataset['validation']['label'], nclasses)

        for fs_mode in ['lasso']:
            print('gamma : ', gamma)
            name = dataset_name + '_' + fs_mode + '_l_' + str(lasso) + '_g_' + str(gamma) + \
                   '_r_' + str(regularization)
            print(name)
            model_kwargs = {
                'lasso': lasso,
                'regularization': regularization
            }
            rank_kwargs = {
                'gamma': gamma,
                'reps': 5
            }
            fit_kwargs = {
                'batch_size': batch_size,
                'epochs': epochs,
                'callbacks': [
                    callbacks.LearningRateScheduler(get_scheduler(epochs))
                ],
                'verbose': 0
            }
            rank = get_rank(fs_mode, data=data, label=label, model_func=create_model,
                              rank_kwargs=rank_kwargs, fit_kwargs=fit_kwargs, model_kwargs=model_kwargs,
                              return_info=False, valid_data=valid_data, valid_label=valid_label)

            for kernel_classifier in kernels_classifier:
                model_kwargs = {
                    'C': 1.,
                    'degree': 3.,
                    'coef0': 1. * (kernel_classifier == 'poly'),
                    'kernel': kernel_classifier,
                    'class_weight': 'balanced',
                    'cache_size': 4096,
                    'max_iter': 10000
                }
                n_features = data.shape[-1]
                classifier_name = fs_mode + '_k_nn_c_' + kernel_classifier
                print(classifier_name)
                nfeats = []
                accuracies = []
                while n_features:
                    n_accuracies = []
                    for _ in range(reps):
                        model = sklearn_SVC(**model_kwargs)
                        model.fit(data[:, rank[:n_features]], np.argmax(label, axis=-1))
                        n_accuracies.append(model.score(valid_data[:, rank[:n_features]], np.argmax(valid_label, axis=-1)))
                        print(
                            'n_features : ', n_features, ', acc : ', n_accuracies[-1]
                        )
                        del model
                    accuracies.append(n_accuracies)
                    nfeats.append(n_features)
                    n_features = int(n_features * gamma_classifier)

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
                        'rank': rank.tolist(),
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

            del rank
