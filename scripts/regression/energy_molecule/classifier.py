from dataset_reader.regression import load_dataset
from keras import backend as K, callbacks
import numpy as np
import json
from scripts.regression import network_models
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

reps = 10
regularization = 0.001
gamma = 0.9
epochs = 60
batch_size = 128

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

dataset_names = ['slice_localization_data']
sd_directory = './scripts/regression/energy_molecule/info/'
dataset_directory = './datasets/regression/'
network_name = 'dense'


def get_factor(n, max_n):
    return 1.5
    # if n < 0.1*max_n:
    #     return 1.5
    # elif n < 0.5*max_n:
    #     return 2.0
    # else:
    #     return 2.5


def scheduler(epoch):
    if epoch < 15:
        return 0.01
    elif epoch < 30:
        return 0.002
    elif epoch < 45:
        return 0.0004
    else:
        return 0.00008


if __name__ == '__main__':
    os.chdir('../../../')
    for dataset_name in dataset_names:
        print('loading dataset', dataset_name)
        dataset = load_dataset(dataset_name, directory=dataset_directory)
        print('STD : ', dataset['std'][-1])
        data = dataset['data']
        result = dataset['result']

        rank_filename = sd_directory + dataset_name + '_' + str(gamma) + '_rank.json'
        result_filename = sd_directory + dataset_name + '_' + str(gamma) + '_result.json'

        try:
            with open(rank_filename) as outfile:
                rank_json = json.load(outfile)
        except:
            continue

        for method in rank_json:
            for example in rank_json[method]:

                total_features = int(np.prod(data.shape[1:]))

                train_data = data[example['train_index']]
                train_result = result[example['train_index']]

                valid_data = data[example['test_index']]
                valid_result = result[example['test_index']]

                print('METHOD : ', method)
                if 'regularization' in example:
                    print('regularization : ', example['regularization'])
                if 'gamma' in example:
                    print('gamma : ', example['gamma'])
                nfeats = []
                results = []
                mses = []
                rank = np.array(example['rank']).astype(int)
                n_features = 10
                # max_features = int(limit * data.shape[-1])
                for factor in [.05, .1, .25, .5]:
                    n_features = int(total_features * factor)
                    r_results = []
                    r_mses = []
                    generator = None
                    if 'dense' in network_name:
                        data_min = np.reshape(train_data, [-1, np.prod(train_data.shape[1:])])[:, rank[:n_features]]
                        valid_data_min = np.reshape(valid_data, [-1, np.prod(valid_data.shape[1:])])[:, rank[:n_features]]
                    else:
                        mask = np.zeros(train_data.shape[1:])
                        mask.flat[rank[:n_features]] = 1.0
                        data_min = data * mask
                        valid_data_min = valid_data * mask
                        generator = dataset['generator']
                    for rep in range(reps):
                        create_model_func = getattr(network_models, network_name.split('_')[0])
                        input_shape = data_min.shape[1:]
                        model = create_model_func(
                            input_shape=input_shape, regularization=regularization
                        )
                        if generator is None:
                            model.fit(data_min, train_result, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[
                                callbacks.LearningRateScheduler(scheduler)
                            ])
                        else:
                            model.fit_generator(
                                generator.flow(data_min, train_result, batch_size=batch_size),
                                steps_per_epoch=len(data_min) // batch_size, epochs=epochs,
                                callbacks=[
                                    # callbacks.ModelCheckpoint(
                                    #     './keras_examples/weights/' + name + '_Weights.h5',
                                    #     monitor="val_acc",
                                    #     save_best_only=True,
                                    #     verbose=1
                                    # ),
                                    callbacks.LearningRateScheduler(scheduler)
                                ],
                                verbose=2
                            )
                        predicted_result = (model.predict(valid_data_min) * dataset['std'][-1] + dataset['mean'][-1]).tolist()
                        r_results.append(predicted_result)
                        mae = model.evaluate(valid_data_min, valid_result, verbose=0)[-1] * dataset['std'][-1]
                        r_mses.append(mae)
                        print('FEATURES : ', n_features, ', MAE : ', mae)

                        del model
                        K.clear_session()
                    print('n_features : ', n_features, ', MAE :', np.mean(r_mses))
                    nfeats.append(n_features)
                    results.append(r_results)
                    mses.append(r_mses)
                    output = {
                        'network': network_name,
                        'n_features': nfeats,
                        'maes': mses
                    }
                mses = np.array(np.mean(mses, axis=-1))
                nfeats = np.array(nfeats)
                roc = 0.5 * (mses[1:] + mses[:-1]) * (nfeats[1:] - nfeats[:-1])
                roc = np.sum(roc) / (nfeats[-1] - nfeats[0])
                print('ROC : ', roc)

                try:
                    with open(result_filename) as outfile:
                        results = json.load(outfile)
                except:
                    results = {}

                if method not in results:
                    results[method] = []

                results[method].append(output)

                with open(result_filename, 'w') as outfile:
                    json.dump(results, outfile)
