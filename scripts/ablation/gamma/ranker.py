from dataset_reader.nips import load_dataset
from src.SFS import get_rank
from src.layers import Mask
from src import saliencies
from src.utils import balance_data
from keras.layers import Activation, Dense, Input, Dropout, BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2, l1
import keras.utils.np_utils as kutils
import keras.backend as K
import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_names = [
    ('arcene', 1e-1),
    ('dexter', 1e-1),
    ('dorothea', 1e-1),
    ('gisette', 1e-1),
    ('madelon', 1e-1),
]

reps = 1
b_size = 100
epochs = 100

root_directory = './scripts/ablation/gamma/info/'
datasets_directory = './datasets/nips/'


def create_model(input_shape, nclasses=2, layer_dims=None, bn=True, kernel_initializer='he_normal',
                 dropout=0.0, lasso=0.0, regularization=0.0, loss_weights=None):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input = Input(shape=input_shape)
    if layer_dims is None:
        layer_dims = [150, 100, 50]
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

    optimizer = optimizers.adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    model.saliency = saliencies.get_saliency('categorical_crossentropy', model)

    return model


if __name__ == '__main__':
    os.chdir('../../../')

    for dataset_name, regularization in dataset_names:

        fs_filename = root_directory + dataset_name + '_ranks.json'

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

        for fs_mode in ['sfs']:
            reps = 1
            lasso = 0.0
            for gamma in [0.0, 0.3, 0.5, 0.75]:
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
                    'reps': 1
                }
                fit_kwargs = {
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'verbose': 0
                }
                rank = get_rank(fs_mode, data=data, label=label, model_func=create_model,
                                rank_kwargs=rank_kwargs, fit_kwargs=fit_kwargs, model_kwargs=model_kwargs)

                try:
                    with open(fs_filename) as outfile:
                        info_data = json.load(outfile)
                except:
                    info_data = {}

                if fs_mode not in info_data:
                    info_data[fs_mode] = []

                info_data[fs_mode].append(
                    {
                        'lasso': lasso,
                        'gamma': gamma,
                        'regularization': regularization,
                        'rank': rank.tolist(),
                        'reps': reps
                    }
                )

                if not os.path.isdir(root_directory):
                    os.makedirs(root_directory)

                with open(fs_filename, 'w') as outfile:
                    json.dump(info_data, outfile)

                del rank
