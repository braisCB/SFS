from keras import backend as K
from keras import initializers, constraints, regularizers
from keras.layers import Layer


class GaussianNoise(Layer):
    """Apply additive zero-centered Gaussian noise.

    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    # Arguments
        stddev: float, standard deviation of the noise distribution.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, stddev, min_value=0., max_value=1.,**kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs, training=None):
        def noised():
            noise_inputs = inputs + K.random_normal(shape=K.shape(inputs),
                                                  mean=0.,
                                                  stddev=self.stddev)
            return K.maximum(K.minimum(noise_inputs, self.max_value), self.min_value)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev, 'max_value': self.max_value, 'min_value': self.min_value}
        base_config = super(GaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Mask(Layer):

    def __init__(self,
                 kernel_initializer='ones',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Mask, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = input_shape[1:]
        self.kernel = self.add_weight(shape=input_dim,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.built = True

    def call(self, inputs, **kwargs):
        output = inputs * self.kernel
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(Mask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
