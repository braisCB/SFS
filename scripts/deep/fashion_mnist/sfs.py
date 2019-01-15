from src.SFS import get_rank
from keras.utils import to_categorical
from keras import callbacks
from keras.datasets import fashion_mnist
from scripts.deep import network_models
import json
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

gamma = 0.
batch_size = 128
epochs = 200
lasso = 0.
regularization = 5e-4
reps = 10

directory = './scripts/deep/fashion_mnist/info_p/'
network_names = ['cnnsimple', 'wrn164']
method = 'sfs'


def scheduler(wrn=True):
    def sch(epoch):
        if not wrn:
            if epoch < 20:
                return .01
            elif epoch < 40:
                return .001
            elif epoch < 60:
                return .0001
            else:
                return .00001
        else:
            if epoch < 40:
                return 0.1
            elif epoch < 80:
                return 0.02
            elif epoch < 115:
                return 0.004
            else:
                return 0.0008

    return sch


rank_kwargs = {
    'reps': 2,
    'gamma': gamma
}


def load_dataset(normalize=True):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    generator = ImageDataGenerator(width_shift_range=4. / 32,
                                   height_shift_range=4. / 32,
                                   fill_mode='nearest',
                                   horizontal_flip=True)
    y_train = np.reshape(y_train, [-1, 1])
    y_test = np.reshape(y_test, [-1, 1])
    if normalize:
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std < 1e-6] = 1.
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    output = {
        'train': {
            'data': x_train,
            'label': y_train
        },
        'test': {
            'data': x_test,
            'label': y_test
        },
        'generator': generator
    }
    return output


def main():

    dataset = load_dataset()

    for network_name in network_names:

        model_func = getattr(network_models, network_name.split('_')[0])

        train_data = np.asarray(dataset['train']['data'])
        train_labels = dataset['train']['label']
        num_classes = len(np.unique(train_labels))

        test_data = np.asarray(dataset['test']['data'])
        test_labels = dataset['test']['label']

        train_labels = to_categorical(train_labels, num_classes=num_classes)
        test_labels = to_categorical(test_labels, num_classes=num_classes)

        epochs = 130 if 'wrn' in network_name else 80

        fit_kwargs = {
            'epochs': epochs,
            'callbacks': [
                callbacks.LearningRateScheduler(scheduler('wrn' in network_name))
            ],
            'verbose': 2
        }

        generator = dataset['generator']
        generator_kwargs = {
            'batch_size': batch_size
        }
        fit_kwargs['steps_per_epoch'] = len(train_data) // batch_size

        print('reps : ', reps)
        print('method : ', method)
        name = 'mnist_' + network_name + '_l_' + str(lasso) + '_g_' + str(gamma) + \
               '_r_' + str(regularization)
        print(name)
        model_kwargs = {
            'nclasses': num_classes,
            'lasso': lasso,
            'regularization': regularization
        }
        saliency_kwargs = {
            'generator': None,
            'horizontal_flip': True
        }
        rank = get_rank(method, data=train_data, label=train_labels,
                        model_func=model_func, model_kwargs=model_kwargs, fit_kwargs=fit_kwargs, generator=generator,
                        generator_kwargs=generator_kwargs, rank_kwargs=rank_kwargs, saliency_kwargs=saliency_kwargs)

        nfeats = []
        accuracies = []
        model_kwargs['lasso'] = 0.
        total_features = int(np.prod(train_data.shape[1:]))
        for factor in [.05, .1, .25, .5]:
            n_features = int(total_features * factor)
            mask = np.zeros(train_data.shape[1:])
            mask.flat[rank[:n_features]] = 1.0
            n_accuracies = []
            for r in range(reps):
                print('factor : ', factor, ' , rep : ', r)
                model = network_models.wrn164(train_data.shape[1:], **model_kwargs)
                model.fit_generator(
                    generator.flow(mask * train_data, train_labels, **generator_kwargs),
                    steps_per_epoch=train_data.shape[0] // batch_size, epochs=130,
                    callbacks=[
                        callbacks.LearningRateScheduler(scheduler(True))
                    ],
                    validation_data=(mask * test_data, test_labels),
                    validation_steps=test_data.shape[0] // batch_size,
                    verbose=2
                )
                n_accuracies.append(model.evaluate(mask * test_data, test_labels, verbose=0)[-1])
                del model
            print(
                'n_features : ', n_features, ', acc : ', n_accuracies
            )
            accuracies.append(n_accuracies)
            nfeats.append(n_features)

        try:
            os.makedirs(directory)
        except:
            pass
        output_filename = directory + network_name + '_' + str(gamma) + '_sfs_results.json'

        try:
            with open(output_filename) as outfile:
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
                'rank': rank.tolist(),
                'reps': reps,
                'classification': {
                    'n_features': nfeats,
                    'accuracy': accuracies
                }
            }
        )

        with open(output_filename, 'w') as outfile:
            json.dump(info_data, outfile)

        del rank


if __name__ == '__main__':
    os.chdir('../../../')
    main()