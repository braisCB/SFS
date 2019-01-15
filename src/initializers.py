from keras import backend as K


def fixed_basis(M):
    def close(shape, dtype=None):
        assert M.shape == shape
        return K.constant(M, dtype=dtype, shape=shape)
    return close
