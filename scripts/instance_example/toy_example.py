from keras import backend as K
from src.SFS import get_saliency_func
from src.layers import Mask
from src import saliency_function
from keras.layers import Activation, Dense, Input, Dropout, BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2, l1
import keras.utils.np_utils as kutils
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

reps = 1
b_size = 100
epochs = 100


def generate_data(nsamples=1000, train=0.7):
    data = np.random.uniform(-1, 1, (nsamples, 5))
    data += 0.25 * np.sign(data)
    y = np.zeros(nsamples)
    pos = np.where(data[:, 0] < 0)[0]
    y[pos] = np.sign(data[pos, 1]) * np.sign(data[pos, 2])
    pos = np.where(data[:, 0] >= 0)[0]
    y[pos] = np.sign(data[pos, 3]) * np.sign(data[pos, 4])
    y[y < 0] = 0
    train_samples = int(nsamples * train)
    return (data[:train_samples], kutils.to_categorical(y[:train_samples], 2)), \
           (data[train_samples:], kutils.to_categorical(y[train_samples:], 2))


def scheduler(epochs):
    if epochs < 60:
        return 0.01
    elif epochs < 120:
        return 0.002
    elif epochs < 160:
        return 0.0004
    else:
        return 0.00008


def create_model(input_shape, nclasses=2, layer_dims=None, bn=True, kernel_initializer='he_normal',
                 dropout=0.0, lasso=1e-2, regularization=1e-3, loss_weights=None):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input = Input(shape=input_shape)
    if layer_dims is None:
        layer_dims = [50]
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

    model.saliency = saliency_function.get_saliency('categorical_crossentropy', model, reduce_func=None, use_abs=False)

    return model


if __name__ == '__main__':
    os.chdir('../../../')
    np.random.seed(42)
    (x_train, y_train), (x_test, y_test) = generate_data(10000)

    model = create_model((5,))
    model.fit(x_train, y_train, batch_size=128, epochs=40, validation_data=(x_test, y_test), verbose=2, callbacks=[LearningRateScheduler(scheduler)])

    p_test_a = np.where(x_test[:, 0] < 0.)[0]
    p_test_b = np.where(x_test[:, 0] >= 0.)[0]

    saliency_a = get_saliency_func(x_test[p_test_a], y_test[p_test_a], model.saliency, batch_size=128, reduce_func=None)
    saliency_b = get_saliency_func(x_test[p_test_b], y_test[p_test_b], model.saliency, batch_size=128, reduce_func=None)

    results = model.predict(x_test)

    print('SALIENCY_A : ', saliency_a.mean(axis=0))
    print('SALIENCY_B : ', saliency_b.mean(axis=0))

    x_toy = .5 * np.array([[-1., -1., -1., -1., 1.], [-1., 1., -1., 1., 1.], [1., 1., -1., -1., -1.], [1., 1., 1., -1., 1.]])
    y_toy = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    saliency = get_saliency_func(x_toy, y_toy, model.saliency, batch_size=128, reduce_func=None)
    print(saliency)
    print(model.predict(x_toy))
