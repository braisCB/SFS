from keras import backend as K
from sklearn.svm import SVC as sklearn_SVC
from sklearn.linear_model import ElasticNet as sklearn_ElasticNet
from keras.layers import Dense, Input, Lambda
from src.kernels import RBF, Poly, Sigmoid, Linear
from src import saliency_function, initializers as custom_initializers
from keras.models import Model
from keras import optimizers
import numpy as np
from scipy.sparse import issparse


class SVC(object):
    def __init__(
            self, nfeatures, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True,
            probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
            decision_function_shape='ovr', random_state=None, saliency_reduce_func='sum', use_keras=True, use_pca=False
    ):
        self.nfeatures = nfeatures
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.class_weight_keras = None
        self.saliency_reduce_func = saliency_reduce_func
        self.use_keras = use_keras
        self.use_pca = use_pca

    def fit(self, X, y, sample_weight=None):
        if isinstance(self.gamma, str):
            if 'auto' in self.gamma:
                self.gamma = 1. / self.nfeatures[0]
            elif self.gamma == 'scale':
                self.gamma = 1. / (self.nfeatures[0] * X.std())
            else:
                raise Exception('gamma ' + self.gamma + ' not supported')
        if isinstance(self.class_weight, str):
            if self.class_weight == 'balanced':
                self.class_weight_keras = y.shape[0] / (y.shape[1] * np.sum(y, axis=0))
            else:
                raise Exception('class_weight ' + self.gamma + ' not supported')
        self.sklearn_model = sklearn_SVC(
            C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0,
            shrinking=self.shrinking, probability=self.probability, tol=self.tol, cache_size=self.cache_size,
            class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape, random_state=self.random_state
        )
        arg_y = np.argmax(y, axis=-1)
        # if self.use_pca:
        #     self.pca = PCA(n_components=max(1, int(self.nfeatures[0]*.75)))
        #     X_ = self.pca.fit_transform(X)
        # else:
        #     X_ = X
        self.sklearn_model.fit(X, arg_y, sample_weight=sample_weight)
        if self.use_keras:
            self.create_keras_model()

    def create_keras_model(self, mask=None):
        landmarks = self.sklearn_model.support_vectors_
        kernel_values = self.sklearn_model.dual_coef_.T
        sparse = issparse(landmarks)
        if sparse:
            landmarks = landmarks.toarray()
            kernel_values = kernel_values.toarray()
        kernel_values = np.concatenate((np.mean(-1. * kernel_values, axis=-1)[:, None], kernel_values), axis=-1)
        beta = self.sklearn_model.intercept_
        beta = np.concatenate(([np.mean(-1. * beta)], beta), axis=-1)
        input = Input(shape=self.nfeatures)
        x = input
        if self.use_pca:
            x = Lambda(lambda t: t - K.constant(self.pca.mean_))(x)
            x = Lambda(lambda t: K.dot(t, K.constant(self.pca.components_.T)))(x)
        if self.kernel == 'linear':
            kernel_svc = Linear(landmarks=landmarks)
        elif self.kernel == 'rbf':
            kernel_svc = RBF(landmarks=landmarks, gamma=self.gamma)
        elif self.kernel == 'poly':
            kernel_svc = Poly(landmarks=landmarks, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        elif self.kernel == 'sigmoid':
            kernel_svc = Sigmoid(landmarks=landmarks, gamma=self.gamma, coef0=self.coef0)
        else:
            raise Exception('kernel ' + str(self.kernel) + ' not supported')
        # x = Mask(kernel_regularizer=l1(0.01), kernel_constraint='NonNeg')(x)
        x = kernel_svc(x, mask=mask)
        kernel_svc.set_weights([landmarks if self.kernel == 'rbf' else landmarks.T])
        nclasses = kernel_values.shape[-1]
        classifier = Dense(
            nclasses, use_bias=True, kernel_initializer='zeros',
            bias_initializer='zeros', input_shape=K.int_shape(x)[-1:]
        )
        output = classifier(x)
        classifier.set_weights([kernel_values, beta])
        self.model = Model(input, output)
        self.output_shape = self.model.output_shape
        self.output = self.model.output
        self.layers = self.model.layers
        self.input = self.model.input
        optimizer = optimizers.SGD(0.1)
        self.model.compile(loss=self.__loss_function('square_hinge'), optimizer=optimizer, metrics=['acc'])
        self.saliency = saliency_function.get_saliency('sklearn_hinge', self.model, reduce_func=self.saliency_reduce_func)
        # self.saliency = saliency_function.get_saliency_gradient('sklearn_hinge', self.model)

    def __loss_function(self, loss_function):
        def loss(y_true, y_pred):
            out = K.relu(1.0 + (1.0 - 2.0*y_true) * y_pred)
            if 'square' in loss_function:
                out = K.square(out)
            if self.class_weight_keras is not None:
                out = self.class_weight_keras * out
            return K.mean(out, axis=-1)
        return loss

    def evaluate(self, X, y, batch_size=None, verbose=1):
        # model_eval = [self.sklearn_model.score(X, y)]
        model_eval = self.model.evaluate(X, y, batch_size=batch_size, verbose=verbose)
        return model_eval

    def predict(self, X, batch_size=None, verbose=1):
        predict = self.model.predict(X, batch_size=batch_size, verbose=verbose)
        return predict

    def __del__(self):
        try:
            del self.saliency
        except:
            pass
        try:
            del self.model
        except:
            pass
        try:
            del self.sklearn_model
        except:
            pass


class ElasticNet(object):
    def __init__(
            self, nfeatures, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False,
            max_iter=10000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None,
            selection='cyclic'
    ):
        self.nfeatures = nfeatures
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def fit(self, X, y, check_input=True):
        self.sklearn_model = sklearn_ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept, normalize=self.normalize,
            precompute=self.precompute, max_iter=self.max_iter, copy_X=self.copy_X, tol=self.tol,
            warm_start=self.warm_start, positive=self.positive, random_state=self.random_state, selection=self.selection
        )
        self.sklearn_model.fit(X, y, check_input=check_input)
        self.create_keras_model()

    def create_keras_model(self):
        kernel_values = self.sklearn_model.coef_.T
        beta = self.sklearn_model.intercept_
        input = Input(shape=self.nfeatures)
        x = input
        # x = Mask(kernel_regularizer=l1(0.01), kernel_constraint='NonNeg')(x)
        nclasses = kernel_values.shape[-1]
        classifier = Dense(
            nclasses, use_bias=self.fit_intercept, kernel_initializer=custom_initializers.fixed_basis(kernel_values),
            bias_initializer=custom_initializers.fixed_basis(beta) if self.fit_intercept else None
        )
        output = classifier(x)
        self.model = Model(input, output)
        self.output_shape = self.model.output_shape
        self.output = self.model.output
        self.layers = self.model.layers
        self.input = self.model.input
        optimizer = optimizers.SGD(0.1)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['acc'])
        self.saliency = saliency_function.get_saliency('mse', self.model)

    def evaluate(self, X, y, batch_size=None, verbose=1):
        model_eval = self.model.evaluate(X, y, batch_size=batch_size, verbose=verbose)
        # arg_y = np.argmax(y, axis=-1)
        # svc_pred = self.sklearn_model.predict(X)
        # arg_pred = np.argmax(svc_pred, axis=-1)
        # pred_mean = (arg_pred == arg_y).mean()
        # print('ours : ', model_eval[-1], ', svc : ', pred_mean)
        return model_eval

    def predict(self, X, batch_size=None, verbose=1):
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model


class KernelElasticNet(object):
    def __init__(
            self, nfeatures, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, alpha=1.0, l1_ratio=0.5,
            fit_intercept=True, normalize=False, precompute=False, max_iter=10000, copy_X=True, tol=0.0001,
            warm_start=False, positive=False, random_state=None, selection='cyclic'
    ):
        self.nfeatures = nfeatures
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection


    def fit(self, X, y, check_input=True):
        if isinstance(self.gamma, str):
            if 'auto' in self.gamma:
                self.gamma = 1. / self.nfeatures[0]
            elif self.gamma == 'scale':
                self.gamma = 1. / (self.nfeatures[0] * X.std())
            else:
                raise Exception('gamma ' + self.gamma + ' not supported')
        landmarks = X
        input = Input(shape=self.nfeatures)
        x = input
        if self.kernel == 'linear':
            x = Linear(landmarks=landmarks)(x)
        elif self.kernel == 'rbf':
            x = RBF(landmarks=landmarks, gamma=self.gamma)(x)
        elif self.kernel == 'poly':
            x = Poly(landmarks=landmarks, gamma=self.gamma, degree=self.degree, coef0=self.coef0)(x)
        elif self.kernel == 'sigmoid':
            x = Sigmoid(landmarks=landmarks, gamma=self.gamma, coef0=self.coef0)(x)
        else:
            raise Exception('kernel ' + str(self.kernel) + ' not supported')
        self.transform_data = Model(input, x)
        X_trans = self.transform_data.predict(X, batch_size=100)
        self.sklearn_model = sklearn_ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept, normalize=self.normalize,
            precompute=self.precompute, max_iter=self.max_iter, copy_X=self.copy_X, tol=self.tol,
            warm_start=self.warm_start, positive=self.positive, random_state=self.random_state, selection=self.selection
        )
        self.sklearn_model.fit(X_trans, y, check_input=check_input)
        self.create_keras_model(X)

    def create_keras_model(self, landmarks):
        kernel_values = self.sklearn_model.coef_.T
        beta = self.sklearn_model.intercept_
        input = Input(shape=self.nfeatures)
        mask = Input(shape=self.nfeatures)
        x = input
        if self.kernel == 'linear':
            x = Linear(landmarks=landmarks)(x, mask)
        elif self.kernel == 'rbf':
            x = RBF(landmarks=landmarks, gamma=self.gamma)(x, mask)
        elif self.kernel == 'poly':
            x = Poly(landmarks=landmarks, gamma=self.gamma, degree=self.degree, coef0=self.coef0)(x, mask)
        elif self.kernel == 'sigmoid':
            x = Sigmoid(landmarks=landmarks, gamma=self.gamma, coef0=self.coef0)(x, mask)
        else:
            raise Exception('kernel ' + str(self.kernel) + ' not supported')
        # x = Mask(kernel_regularizer=l1(0.01), kernel_constraint='NonNeg')(x)
        nclasses = kernel_values.shape[-1]
        classifier = Dense(
            nclasses, use_bias=self.fit_intercept, kernel_initializer=custom_initializers.fixed_basis(kernel_values),
            bias_initializer=custom_initializers.fixed_basis(beta) if self.fit_intercept else None
        )
        output = classifier(x)
        self.model = Model([input, mask], output)
        self.output_shape = self.model.output_shape
        self.output = self.model.output
        self.layers = self.model.layers
        self.input = self.model.input
        optimizer = optimizers.SGD(0.1)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['acc'])
        self.saliency = saliency_function.get_saliency('mse', self.model)

    def evaluate(self, X, y, mask=None, batch_size=None, verbose=1):
        if mask is None:
            mask = np.ones_like(X)
        model_eval = self.model.evaluate([X, mask], y, batch_size=batch_size, verbose=verbose)
        # arg_y = np.argmax(y, axis=-1)
        # svc_pred = self.sklearn_model.predict(X)
        # arg_pred = np.argmax(svc_pred, axis=-1)
        # pred_mean = (arg_pred == arg_y).mean()
        # print('ours : ', model_eval[-1], ', svc : ', pred_mean)
        return model_eval

    def predict(self, X, mask=None, batch_size=None, verbose=1):
        if mask is None:
            mask = np.ones_like(X)
        return self.model.predict([X, mask], batch_size=batch_size, verbose=verbose)

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
