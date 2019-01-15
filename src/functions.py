from keras.layers import Input, Activation
from keras.models import Model


def fs_add_to_model(fs_layer, model, input_shape, activation=None):
    # input_shape = fs_layer.input_shape
    input = Input(shape=input_shape)
    x = fs_layer(input)
    if activation is not None:
        x = Activation(activation=activation)(x)
    output = model(x)
    model = Model(input, output)
    return model