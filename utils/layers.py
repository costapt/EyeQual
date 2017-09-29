"""Specializaed layers."""
import numpy as np

from keras import activations
from keras import constraints
from keras import regularizers
from keras import initializations
from keras.engine import Layer, InputSpec

from keras import backend as K


class WeightedAveragePooling(Layer):

    def __init__(self, shape, **kwargs):
        self.shape = shape
        super(WeightedAveragePooling, self).__init__(**kwargs)

    def build(self, input_shape):
        init = initializations.get('one')
        self.alpha = self.add_weight(shape=self.shape, initializer=init,
                                     name='alpha')

    def call(self, x, mask=None):
        alpha_sig = K.abs(self.alpha)
        alpha_batch = K.repeat_elements(alpha_sig, K.shape(x)[0], 0)
        normed_x = x * alpha_batch
        sum_x = K.expand_dims(K.sum(normed_x, axis=(1, 2, 3)))
        sum_alpha = K.expand_dims(K.sum(alpha_batch, axis=(1, 2, 3)))
        return sum_x / (sum_alpha + K.epsilon())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        config = {
            'shape': self.shape,
        }
        base_config = super(WeightedAveragePooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SWAP(Layer):

    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(SWAP, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            w, b = self.get_weights()
            w = w / np.sum(np.absolute(w))
            self.set_weights([w, b])
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        output = K.dot(x, K.abs(self.W))
        if self.bias:
            output -= 0.5 * np.ones((1,))
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(SWAP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
