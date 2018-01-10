import theano
import lasagne

from theano import tensor as T
import numpy as np
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import layers as ll

srng = RandomStreams(seed=1)

def clipped_binary_crossentropy(predictions, targets):
    targets = T.clip(targets, 0, 1)
    return T.nnet.binary_crossentropy(predictions, targets)


class RecognitionModel(object):
    """
    Recognition Model Interface Class
    The recognition model approximates the posterior given observed fluorescence: q_phi(z|x).
    """

    def __init__(self, recognition_params, X, S, P, rng, batch_size):
        """
        :param X: Symbolic variable for the traces
        :param S: Symbolic variable for the inferred spikes
        :param P: Symbolic variable for inferred parameters
        :param rng: RandomNumber generator shared across the whole model to guarantee reproducability
        :param batch_size: Batch size
        """

        self.X = X
        self.S = S
        self.P = P
        self.batch_size = batch_size
        self.rng = rng
        self.pz = 0.015

        self.p_sigma = recognition_params['p_sigma']
        self.forw_backw = recognition_params['forw_backw']
        self.n_genparams = recognition_params['n_genparams']

    def eval_prior(self, buffers):
        """
        Evaluates prior on the latent variables
        """
        zsamp = self.S[:, buffers[0]:-buffers[1]]
        n_samples = zsamp.shape[0] // self.batch_size
        zsamp = zsamp.reshape((self.batch_size, n_samples, -1))
        ks = zsamp.sum(axis=-1)
        ns = zsamp.shape[-1].astype(config.floatX) * T.ones_like(ks)
        log_nok = T.gammaln(ns + 1) - T.gammaln(ks + 1) - T.gammaln(ns - ks + 1)
        log_p = 0
        if self.n_genparams == 1:
            log_p = -0.5 * (T.log(2 * np.pi) + 2 * T.log(self.p_sigma) + ((self.P / self.p_sigma) ** 2).sum(axis=-1))
            log_p = log_p.reshape((self.batch_size, n_samples))

        return log_nok + ks * T.log(self.pz) + (ns - ks) * T.log(1 - self.pz) + log_p

    def get_sample(self):
        """
        Evaluates posterior and draws samples
        """
        pass

    def get_params(self):
        """
        Returns all (trainable) parameters of the Recognition network
        """
        pass

    def eval_log_density(self):
        """
        Evaluate the the density q_phi(z_n|x) for a given sample
        """
        pass


class BernoulliRecognition(RecognitionModel):
    def __init__(self, recognition_params, X, S, P, rng, batch_size=1):
        """
        Factorized posterior using just the convolutional network
        """
        super().__init__(recognition_params, X, S, P, rng, batch_size)
        self.NN = recognition_params['network']

        self.p = ll.get_output(self.NN, inputs=self.X)
        sample = srng.binomial(self.p.shape, n=1, p=self.p, dtype=theano.config.floatX)

        self.recfunc = theano.function([self.X], outputs={'Probs':self.p,'Spikes':sample})

    def get_params(self):
        network_params = ll.get_all_params(self.NN, trainable=True)
        return network_params

    def get_sample(self, x, n_samples=1):

        p = self.recfunc(x)['Probs']
        p = np.repeat(p, n_samples, axis=0)
        sample = self.rng.binomial(n=1, p=p).astype(config.floatX)

        return {'Probs':p,'Spikes':sample}

    def eval_log_density(self, hsamp, buffers=(0, 1), for_eval=False):

        batch_size = self.batch_size
        n_samples = hsamp.shape[0] // batch_size
        if for_eval:
            batch_size = hsamp.shape[0]
            n_samples = 1

        prob = T.clip(self.p, 0.001, 0.999)
        prob = prob.dimshuffle(0, 'x', 1)
        hsamp = hsamp.reshape((batch_size, n_samples, -1))
        return -clipped_binary_crossentropy(prob, hsamp)[:, :, buffers[0]:-buffers[1]].sum(axis=-1)


class GRUX_BernoulliRecognition(RecognitionModel):
    """
    This architecture adds additional recurrent layers on top of the convolutional network to obtain a correlated  posterior. 
    The posterior probability is evaluated at each timestep, and a sample from that timestep is drawn and fed back into the RNN.
    """
    def __init__(self, recognition_params, X, S, P, rng, batch_size=1):

        super().__init__(recognition_params, X, S, P, rng, batch_size)
        ret_dict = {}

        self.n_units = recognition_params['rnn_units']
        self.n_convfeatures = recognition_params['n_features']

        self.init_rnn_forw = theano.shared(np.zeros([1, self.n_units]).astype(config.floatX))
        self.init_rnn_back = theano.shared(np.zeros([1, self.n_units]).astype(config.floatX))

        n_inputs = self.n_convfeatures + 1 + 1 + self.n_units * int(self.forw_backw)

        self.conv_layer = recognition_params['network']
        self.conv_output = ll.get_output(self.conv_layer, inputs=self.X)

        x_inp = recognition_params['input']
        x_inp.input_var = self.X
        trace_layer = ll.DimshuffleLayer(x_inp, (0, 1, 'x'))
        self.trace_output = ll.get_output(trace_layer)

        gru_cell_inp = ll.InputLayer((None, n_inputs))
        self.init_rnn_forw_layer = ll.InputLayer((None, self.n_units), input_var=self.init_rnn_forw.repeat(self.X.shape[0] // self.init_rnn_forw.shape[0], 0))
        self.gru_layer_inp = ll.InputLayer((None, None, n_inputs))
        self.gru_cell = ll.GRUCell(gru_cell_inp, self.n_units, grad_clipping=10., name='forw_rnn', hid_init=self.init_rnn_forw_layer)
        self.gru_layer = ll.RecurrentContainerLayer({gru_cell_inp: self.gru_layer_inp}, self.gru_cell['output'])
        self.p_layer = ll.DenseLayer((None, self.n_units + n_inputs - 2), 1, nonlinearity=lasagne.nonlinearities.sigmoid, b=lasagne.init.Constant(-5.), name='dense_output')

        hid_0 = self.init_rnn_forw.repeat(self.X.shape[0] // self.init_rnn_forw.shape[0], 0)
        samp_0 = T.zeros([self.X.shape[0], 1])

        scan_inp = T.concatenate([self.conv_output, self.trace_output], axis=2)

        if self.forw_backw:
            inp_back = ll.ConcatLayer([self.conv_layer, trace_layer], axis=2)
            init_rnn_back_layer = ll.InputLayer((None, self.n_units), input_var=self.init_rnn_back.repeat(self.X.shape[0] // self.init_rnn_back.shape[0], 0))
            self.back_layer = ll.GRULayer(inp_back, self.n_units, backwards=True, name='back_rnn', hid_init=init_rnn_back_layer)
            self.back_output = ll.get_output(self.back_layer)

            scan_inp = T.concatenate([scan_inp, self.back_output], axis=2)

        def sample_step(scan_inp_, hid_tm1, samp_tm1):

            cell_in = T.concatenate([scan_inp_, samp_tm1], axis=1)
            rnn_t_ = self.gru_cell.get_output_for({'input': cell_in, 'output': hid_tm1})
            prob_in = T.concatenate([cell_in[:, :-2], rnn_t_['output']], axis=1)
            prob_t = self.p_layer.get_output_for(prob_in)
            samp_t = srng.binomial(prob_t.shape, n=1, p=prob_t, dtype=theano.config.floatX)

            return rnn_t_['output'], samp_t, prob_t

        ((rnn_t, s_t, p_t), updates) = \
            theano.scan(fn=sample_step,
                        sequences=[scan_inp.dimshuffle(1, 0, 2)],  # Scan iterates over first dimension, so we have to put time in front.
                        outputs_info = [hid_0, samp_0, None, ])

        ret_dict['Probs'] = p_t[:, :, 0].T
        ret_dict['Spikes'] = s_t[:, :, 0].T

        if self.n_genparams:
            self.gen_mu_layer = ll.DenseLayer((None, None, self.n_units), self.n_genparams, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Normal(std=0.01, mean=0.0), num_leading_axes=2,
                                              name='dense_gen_mu')
            self.gen_sig_layer = ll.DenseLayer((None, None, self.n_units), self.n_genparams, nonlinearity=lasagne.nonlinearities.softplus, W=lasagne.init.Normal(std=0.01, mean=-0.0),
                                               b=lasagne.init.Constant(-2.), num_leading_axes=2, name='dense_gen_sig')

            ret_dict['Gen_mu'] = self.gen_mu_layer.get_output_for(rnn_t).mean(0)
            ret_dict['Gen_sig'] = self.gen_sig_layer.get_output_for(rnn_t).mean(0)

        self.recfunc = theano.function([self.X], outputs=ret_dict, updates=updates, on_unused_input='ignore')

    def get_params(self):
        network_params = ll.get_all_params(self.conv_layer, trainable=True)
        for p in ll.get_all_params(self.gru_layer):
            network_params.append(p)
        for p in ll.get_all_params(self.p_layer):
            network_params.append(p)
        if self.forw_backw:
            for p in ll.get_all_params(self.back_layer)[-10:]:
                network_params.append(p)
        if self.n_genparams:
            for p in ll.get_all_params(self.gen_mu_layer):
                network_params.append(p)
            for p in ll.get_all_params(self.gen_sig_layer):
                network_params.append(p)
        return network_params

    def get_sample(self, x, n_samples=1):

        x = np.repeat(x, n_samples, axis=0)

        ret_dict = self.recfunc(x)

        if self.n_genparams == 1:
            p_sample = np.random.normal(ret_dict['Gen_mu'][:, 0], ret_dict['Gen_sig'][:, 0]).astype(config.floatX)
            ret_dict['Params'] = np.expand_dims(p_sample, 1)
        return ret_dict

    def eval_log_density(self, hsamp, buffers=(0, 1), for_eval=False):

        batch_size = self.batch_size
        n_samples = hsamp.shape[0] // batch_size
        if for_eval:
            batch_size = hsamp.shape[0]
            n_samples = 1

        sample_output = T.concatenate((T.zeros(hsamp[:, :1].shape), hsamp[:, :-1]), axis=1)
        sample_output = T.reshape(sample_output, [sample_output.shape[0], sample_output.shape[1], 1])

        conv_output = T.repeat(self.conv_output, n_samples, axis=0)
        trace_output = T.repeat(self.trace_output, n_samples, axis=0)

        gru_inp = T.concatenate([conv_output, trace_output], axis=2)

        if self.forw_backw:
            back_output = T.repeat(self.back_output, n_samples, axis=0)
            gru_inp = T.concatenate([gru_inp, back_output], axis=2)

        gru_inp = T.concatenate([gru_inp, sample_output], axis=2)
        rnn_t = ll.get_output(self.gru_layer, inputs={self.gru_layer_inp: gru_inp, self.init_rnn_forw_layer: self.init_rnn_forw.repeat(hsamp.shape[0] // self.init_rnn_forw.shape[0], 0)})

        rnn_flat = T.concatenate([gru_inp[:, :, :-2], rnn_t], axis=2)
        rnn_flat = rnn_flat.reshape([-1, rnn_flat.shape[-1]])

        prob = self.p_layer.get_output_for(rnn_flat)
        prob = T.clip(prob, 0.001, 0.999)
        prob = prob.reshape([batch_size, n_samples, -1])

        hsamp = hsamp.reshape((batch_size, n_samples, -1))

        log_density = -clipped_binary_crossentropy(prob, hsamp)[:, :, buffers[0]:-buffers[1]].sum(axis=-1)

        if self.n_genparams and not for_eval:
            gen_mu = self.gen_mu_layer.get_output_for(rnn_t.dimshuffle(1, 0, 2)).mean(0)
            gen_sig = self.gen_sig_layer.get_output_for(rnn_t.dimshuffle(1, 0, 2)).mean(0)

            p_logl = -0.5 * (T.log(2 * np.pi) + 2 * T.log(gen_sig) + ((self.P - gen_mu) / gen_sig) ** 2)
            p_logl = p_logl.reshape((batch_size, n_samples))

            log_density = log_density + p_logl

        return log_density

