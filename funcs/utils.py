import numpy as np
import theano.tensor as T
from theano import config

import sys
sys.path.append('../funcs/')
import layers as ll
import lasagne.nonlinearities as lnl
import lasagne


def binary_crossentropy(output, target):
    """Clipped binary crossentropy"""
    target = np.clip(target,0,1)
    output = np.clip(output,0.001, 0.999)
    return target * np.log(output) + (1.0 - target) * np.log(1.0 - output)


def rebin(arr, factor, method = 'sum'):
    """Rebins a spike train by a given factor, shortens the array to enable division"""
    if np.ndim(arr) == 1: arr = np.expand_dims(arr, axis=0)

    arr = arr[:,:(arr.shape[1]//factor)*factor]
    if method == 'sum': return arr.reshape((arr.shape[0],arr.shape[1]//factor,factor)).sum(-1)
    else: return arr.reshape((arr.shape[0],arr.shape[1]//factor,factor)).mean(-1)


def softp(x):
    return (np.log(1+np.exp(x)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inv_softp(x):
    return np.log(np.exp(x)-1)


def inv_sigmoid(x):
    return -np.log(1/x-1)


def set_rec_net(num_filters, filtsize, nonlin=lnl.LeakyRectify(0.2), AR = False, FB = False, n_rnn_units=64, n_features=13, n_genparams=0, p_sigma=None):
    """
    :param num_filters: Number of filters in each layer for the convolutional network, also sets number of layers
    :param filtsize: Size of the filters in each layer
    :param nonlin: Nonlinearity used
    
    These parameters are only relevant if a RNN is used to obtain a correlated posterior estimation
    
    :param AR: Whether this network is used as the first stage of an auto-regressive network
    :param FB: Whether the auto-regressive network uses a backwards running RNN
    :param n_rnn_units: Number of units in the RNN
    :param n_features: Number of features passed form the CNN to the RNN
    :param n_genparams: Number of generative model parameters inferred by the recognition network
    :param p_sigma: standard deviation of the prior on the inffered generative model parameter.
    """
    input_l = ll.InputLayer((None, None))
    rec_nn = ll.DimshuffleLayer(input_l, (0, 'x', 1))
    hevar = np.sqrt(np.sqrt(2/(1+0.2**2)))
    convout_nonlin = nonlin
    if n_features == 1: convout_nonlin = lnl.linear

    if nonlin == lnl.tanh:
        init = lasagne.init.GlorotUniform()
    else:
        init = lasagne.init.HeNormal(hevar)

    for num, size in zip(num_filters,filtsize):

        rec_nn = (ll.Conv1DLayer(rec_nn, num_filters=num, filter_size=size, stride =1, pad='same', nonlinearity=nonlin, name='conv_filter', W=init))

    if not AR:
        prob_nn = (ll.Conv1DLayer(rec_nn, num_filters=1, filter_size=11,stride =1, pad='same', nonlinearity=lnl.sigmoid,name='conv_out', b=lasagne.init.Constant(-3.)))
        prob_nn = ll.DimshuffleLayer(prob_nn,(0,2,1))
        prob_nn = ll.FlattenLayer(prob_nn)
    else:
        prob_nn = (ll.Conv1DLayer(rec_nn, num_filters=n_features,filter_size=11,stride =1, pad='same', nonlinearity=convout_nonlin,name='conv_out'))
        prob_nn = ll.DimshuffleLayer(prob_nn,(0,2,1))
        if n_features == 1: prob_nn = ll.FlattenLayer(prob_nn)

    return {'network': prob_nn, 'input': input_l, 'n_features': n_features, 'rnn_units': n_rnn_units, 'n_genparams': n_genparams, 'p_sigma': p_sigma, 'forw_backw': FB}