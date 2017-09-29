"""The model definitions."""
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Flatten
from keras.regularizers import WeightRegularizer
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from utils.layers import WeightedAveragePooling, SWAP


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    return Convolution2D(f, k, k, border_mode=border_mode, subsample=(s, s), **kwargs)


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    return BatchNormalization(mode=mode, axis=axis, **kwargs)


def ConvBlock(i, nf, k=3, s=1, border_mode='same', maxpool=True, norm=True, **kwargs):
    """A Conv-Pool-LeakyRelu-Batchnorm block."""
    x = Convolution(nf, k=k, s=s, border_mode=border_mode, **kwargs)(i)
    if maxpool:
        x = MaxPooling2D((2, 2))(x)

    x = LeakyReLU(0.02)(x)

    if norm:
        x = BatchNorm()(x)

    return x


def micnn(nf, input_size=512, n_blocks=4):
    """
    The micnn model.

    The model extracts features from the input. Then we just need to run a
    1x1 Convolution on top of these features to get a heatmap.
    """
    img = Input(shape=(3,) + (input_size, input_size))

    ###########################################################################
    #                           MAIN NETWORK DEFINITION                       #
    ###########################################################################
    xi = img
    for i in range(n_blocks):
        nfi = nf * 2**i
        if nfi > nf * 8:
            nfi = nf * 8

        xi = ConvBlock(xi, nfi, maxpool=True, norm=True)
    ###########################################################################
    #                           FINAL CLASSIFICATIONS                         #
    ###########################################################################
    xi = ConvBlock(xi, nfi, k=1, s=1, maxpool=False)

    return img, xi


def get_out_size(input_size, n_blocks):
    return input_size / (2**n_blocks)


def quality_assessment(nf, l2=0, input_size=512, n_blocks=4, lr=2e-4,
                       pooling='SWAP', pooling_wreg=1, pooling_breg=1e-1):
    """EyeQual implementation."""
    out_size = get_out_size(input_size, n_blocks)
    img, xi = micnn(nf, input_size=input_size, n_blocks=n_blocks)
    quality_map = Convolution(1, k=1, s=1, activation='sigmoid')(xi)

    if pooling == 'SWAP':
        out = Flatten()(quality_map)
        out = SWAP(1, activation='sigmoid', W_regularizer=WeightRegularizer(l2=pooling_wreg),
                   init='one', b_regularizer=WeightRegularizer(l2=pooling_breg),
                   name='pool')(out)
    elif pooling == 'WAP':
        out = WeightedAveragePooling((1, 1, out_size, out_size), name='pool')(quality_map)
    elif pooling == 'AP':
        out = AveragePooling2D((out_size, out_size))(quality_map)
        out = Flatten()(out)
    elif pooling == 'MP':
        out = MaxPooling2D((out_size, out_size))(quality_map)
        out = Flatten()(out)

    quality_model = Model(img, quality_map)
    model = Model(img, out)

    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model, quality_model
