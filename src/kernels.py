from keras import backend as K
from keras.engine import Layer


class RBF(Layer):

    def __init__(self, landmarks=None, gamma='auto', **kwargs):
        if landmarks is None:
            raise Exception('landmarks must be placed')
        self.output_dim = landmarks.shape[0]
        if gamma == 'auto':
            self.gamma = K.constant(1.0 / landmarks.shape[1])
        else:
            self.gamma = K.constant(gamma)
        self.landmarks_ = landmarks.shape
        self.supports_masking = True
        super(RBF, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RBF, self).build(input_shape)
        self.landmarks = self.add_weight(shape=self.landmarks_,
                                                initializer='zeros',
                                                name='landmarks',
                                                trainable=False
                                                )
        self.built = True

    def call(self, x, mask=None, **kwargs):
        def dist(v):
            diff = self.landmarks - v
            if mask is not None:
                diff *= mask
            return K.sum(diff * diff, axis=-1)
        D = K.map_fn(dist, x)
        return K.exp(-self.gamma * D)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Poly(Layer):

    def __init__(self, landmarks=None, degree=1, gamma='auto', coef0=0.0, **kwargs):
        if landmarks is None:
            raise Exception('landmarks must be placed')
        self.output_dim = landmarks.shape[0]
        self.coef0 = coef0
        if gamma == 'auto':
            self.gamma = K.constant(1.0 / landmarks.shape[1])
        else:
            self.gamma = K.constant(gamma)
        self.degree = K.constant(degree)
        self.landmarks_ = landmarks.T.shape
        self.supports_masking = True
        super(Poly, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Poly, self).build(input_shape)
        self.landmarks = self.add_weight(shape=self.landmarks_,
                                                initializer='zeros',
                                                name='landmarks',
                                                trainable=False
                                         )
        self.built = True

    def call(self, x, mask=None, **kwargs):
        if mask is not None:
            x *= mask
        H = self.gamma * K.dot(x, self.landmarks) + self.coef0
        return K.pow(H, self.degree)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(Poly, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Linear(Layer):

    def __init__(self, landmarks=None, **kwargs):
        if landmarks is None:
            raise Exception('landmarks must be placed')
        self.output_dim = landmarks.shape[0]
        self.landmarks_ = landmarks.T.shape
        self.supports_masking = True
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Linear, self).build(input_shape)
        self.landmarks = self.add_weight(shape=self.landmarks_,
                                                initializer='zeros',
                                                name='landmarks',
                                                trainable=False
                                                )
        self.built = True

    def call(self, x, mask=None, **kwargs):
        if mask is not None:
            x *= mask
        H = K.dot(x, self.landmarks)
        return H

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sigmoid(Layer):

    def __init__(self, landmarks=None, gamma='auto', coef0=0.0, **kwargs):
        if landmarks is None:
            raise Exception('landmarks must be placed')
        self.output_dim = landmarks.shape[0]
        self.coef0 = coef0
        if gamma == 'auto':
            self.gamma = K.constant(1.0 / landmarks.shape[1])
        else:
            self.gamma = K.constant(gamma)
        self.landmarks_ = landmarks.T.shape
        self.supports_masking = True
        super(Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sigmoid, self).build(input_shape)
        self.landmarks = self.add_weight(shape=self.landmarks_,
                                                initializer='zeros',
                                                name='landmarks',
                                                trainable=False
                                        )
        self.built = True

    def call(self, x, mask=None, **kwargs):
        if mask is not None:
            x *= mask
        H = self.gamma * K.dot(x, self.landmarks) + self.coef0
        return K.tanh(H)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
