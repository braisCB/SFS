from keras.models import Model
from keras import backend as K, optimizers
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Input, Flatten
from keras.regularizers import l1, l2
from src.layers import Mask
from src import saliencies


def dense(input_shape, layer_dims=None, bn=True, kernel_initializer='he_normal',
                 dropout=0.0, lasso=0.0, regularization=0.0):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input = Input(shape=input_shape)
    if layer_dims is None:
        layer_dims = [150, 100, 50]
    try:
        x = Flatten()(input)
    except:
        x = input
    if lasso > 0:
        x = Mask(kernel_regularizer=l1(lasso))(x)

    for layer_dim in layer_dims:
        x = Dense(layer_dim, use_bias=not bn, kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2(regularization) if regularization > 0.0 else None)(x)
        if bn:
            x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        x = Activation('relu')(x)

    x = Dense(1, use_bias=True, kernel_initializer=kernel_initializer,
              kernel_regularizer=l2(regularization) if regularization > 0.0 else None)(x)
    output = x

    model = Model(input, output)

    optimizer = optimizers.adam(lr=1e-3)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    model.saliency = saliencies.get_saliency('mean_squared_error', model)

    return model


