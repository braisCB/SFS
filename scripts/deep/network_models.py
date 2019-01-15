from keras.models import Model
from keras import backend as K, optimizers
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Input, Flatten, \
    Convolution2D, MaxPool2D, GlobalAveragePooling2D
from keras.regularizers import l1, l2
from src.layers import Mask, GaussianNoise
from src import saliencies
from scripts.deep.wide_residual_network import wrn_block


def cnnsimple(input_shape, nclasses=2, bn=True, kernel_initializer='he_normal',
                 dropout=0.0, lasso=0.0, regularization=0.0):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input = Input(shape=input_shape)
    layer_dims = [16, 32]

    x = Mask(kernel_regularizer=l1(lasso))(input) if lasso > 0 else input

    for layer_dim in layer_dims:
        x = Convolution2D(layer_dim, (3,3), padding='same', use_bias=not bn, kernel_initializer=kernel_initializer,
                          kernel_regularizer=l2(regularization) if regularization > 0.0 else None)(x)
        if bn:
            x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        x = Activation('relu')(x)
        x = MaxPool2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(1024, use_bias=not bn, kernel_initializer=kernel_initializer,
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

    model.saliency = saliencies.get_saliency('categorical_crossentropy', model)

    return model


def wrn164(
    input_shape, nclasses=2, bn=True, kernel_initializer='he_normal', dropout=0.0, lasso=0.0, regularization=0.0,
    input_noise=0.
):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    ip = Input(shape=input_shape)

    x = Mask(kernel_regularizer=l1(lasso))(ip) if lasso > 0 else ip

    if input_noise:
        x = GaussianNoise(stddev=input_noise)(x)

    x = Convolution2D(
        16, (3, 3), padding='same', kernel_initializer=kernel_initializer,
        use_bias=False, kernel_regularizer=l2(regularization) if regularization > 0.0 else None
    )(x)

    l = 16
    k = 4

    output_channel_basis = [16, 32, 64]
    strides = [1, 2, 2]

    N = (l - 4) // 6

    for ocb, stride in zip(output_channel_basis, strides):
        x = wrn_block(
            x, ocb * k, N, strides=(stride, stride), dropout=dropout,
            regularization=regularization, kernel_initializer=kernel_initializer
        )

    x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
    x = Activation('relu')(x)

    deep_features = GlobalAveragePooling2D()(x)

    classifier = Dense(nclasses, kernel_initializer=kernel_initializer)

    output = classifier(deep_features)
    output = Activation('softmax')(output)

    model = Model(ip, output)

    optimizer = optimizers.SGD(lr=1e-1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    model.saliency = saliencies.get_saliency('categorical_crossentropy', model)

    return model


