import theano
import theano.tensor as T
from theano import In

import numpy as np
from theano import config

def softp(x):
    return np.log(1 + np.exp(x))


class GenerativeModel(object):
    """
    Generative Model Interface Class
    The generative model describes the fluorescence dynamics for given latent spikes: p_theta(x|z).
    """
    def __init__(self, init_params, inf_params=None, X=T.matrix(), S=T.matrix(), P=T.matrix(), batch_size=1, rng=None):

        """
        :param init_params: The initial values of the generative model.
        :param inf_params: Params that are inferred by the recognition model.
        :param X: Symbolic variable for the traces
        :param S: Symbolic variable for the inferred spikes
        :param P: Symbolic variable for inferred parameters
        :param batch_size: Batchsize
        :param rng: RandomNumber generator shared across the whole model to guarantee reproducability
        """

        self.X = X
        self.S = S
        self.P = P

        self.batch_size = batch_size
        self.fixed_gen = False
        self.true_params = None
        self.init_params = init_params

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

        self.trainable_pars = []

        self.pars = {}
        if not inf_params: self.inf_params = {}
        else: self.inf_params = inf_params
        n_infs = 0
        for k in sorted(init_params):
            if k in self.inf_params:
                self.pars[k] = self.P[:, n_infs] + init_params[k]
                n_infs += 1
            else:
                self.pars[k] = theano.shared(init_params[k] * np.ones([1], dtype=config.floatX), broadcastable=[True], name=k)
                if k is not 'delta_t': self.trainable_pars.append(self.pars[k])

    def get_params(self):
        """
        Return the trainable parameters of the generative model
        """
        return self.trainable_pars

    def set_params(self, params):
        """
        Set parameters with float values
        :param params: Dict of parameters
        """
        for k in sorted(params):
            one = self.pars[k].get_value()/self.pars[k].get_value()
            self.pars[k].set_value(params[k]*one)

    def load_genparams(self, c):
        """
        Swaps in parameters of a cell
        :param c: Cell number
        """
        for p in self.get_params():
            p.set_value(self.genparam_dicts[c][p.name])

    def print_params(self):
        """
        Prints parameter values of current cell        
        """
        for p in self.get_params():
            print(p, ': ', np.round(np.max(p.eval()), 2))

    def store_genparams(self, c):
        """
        Writes parameters of the currently active cell into a dict
        :param c: Cell number
        """
        for p in self.get_params():
            self.genparam_dicts[c][p.name] = p.get_value(return_internal_type=True)


    def eval_log_density(self, buffers=None, bl=None):
        """
        Function evaluating the likelihood of the reconstruction given the latents: p_theta(x|z).
        :param buffers: Buffers on the left and right side, these segments are not evaluated and don't influence the gradients. 
        :param bl: Whether a baseline estimation is used. If yes it gets subtracted from the trace, so that the generative model only tries to reconstruct the transients that are actually
        cause by spikes.
        :return: Likelihood
        """

        prediction = self.F
        if not buffers: buffers = self.buffers

        n_samples = T.shape(self.S)[0] // self.batch_size
        prediction = T.reshape(prediction, (self.batch_size, n_samples, -1))
        target_var = self.X.dimshuffle(0, 'x', 1)

        if bl:
            bl = T.reshape(self.BL, (self.batch_size, n_samples, -1))
            target_var = target_var - bl

        superres = (prediction.shape[-1] / target_var.shape[-1]).astype('int32')
        prediction = prediction[:, :, ::superres]

        sig = softp(self.pars['sigma'])

        return (-(0.5 * T.log(2 * np.pi) + T.log(sig)) - 0.5 * ((prediction - target_var) / sig) ** 2)[:, :, buffers[0]:-buffers[1]].sum(axis=-1)


class FOOPSI(GenerativeModel):
    """
    Simple linear model as described in the paper.
    Sigmoid and softplus functions are used to ensure valid parameters. Otherwise training might crash.
    
    C_(t) = s + sigmoid(g) * C_(t-1)
    F_(t) = softplus(α) * C_(t) + σ
    """
    def __init__(self, init_params, inf_params=None, X=T.matrix(), S=T.matrix(), P=T.matrix(), batch_size=1, rng=None):
        super().__init__(init_params, inf_params, X, S, P, batch_size, rng)

        self.type = 'FOOPSI'
        start = T.zeros([self.S.shape[0]])

        def one_step(s, C_tm1, g):
            C = s + T.nnet.sigmoid(g) * C_tm1
            return C

        C, updates = theano.scan(fn=one_step,
                                 sequences=[self.S.T],
                                 outputs_info=[start],
                                 non_sequences=[self.pars['gamma']])

        self.C = C.T
        self.F = (softp(self.pars['alpha']) * self.C.T + self.pars['beta']).T
        self.genfunc = theano.function([self.S, In(self.P, value=np.zeros([1, 1]).astype(config.floatX))], self.F, on_unused_input='ignore')


class SCDF(GenerativeModel):
    """
    Nonlinear model as described in the paper, allows for facilitation and saturation.
    Sigmoid and softplus functions are used to ensure valid parameters. Otherwise training might crash.
    min(D_(t-1) , D_max) ensures that dye concentration stays below the max value, this prevents oscillations that can blow up.
    
    C_(t) = s + sigmoid(γ) * C_(t-1)
    D_(t) = softplus(η) * (C_(t) ^ (1 + softplus(ζ))) * (D_max - min(D_(t-1) , D_max)) + sigmoid(κ) * min(D_(t-1) , D_max)
    F_(t) = softplus(α) * D_(t) + β + sigmoid(σ)
    """
    def __init__(self, init_params, inf_params=None, X=T.matrix(), S=T.matrix(), P=T.matrix(), batch_size=1, rng=None):
        super().__init__(init_params, inf_params, X, S, P, batch_size, rng)

        self.type = 'SCDF'
        start = T.zeros([self.S.shape[0]])

        def one_step(s, C_tm1, D_tm1, g, eta, zeta, D_m, kappa):
            C = s + T.nnet.sigmoid(g) * C_tm1
            D = softp(eta) * (C ** (1 + softp(zeta))) * (D_m - T.minimum(D_tm1, D_m)) + T.nnet.sigmoid(kappa) * T.minimum(D_tm1, D_m)
            return C, D

        C_D, updates = theano.scan(fn=one_step,
                                   sequences=[self.S.T],
                                   outputs_info=[start, start],
                                   non_sequences=[self.pars['gamma'], self.pars['eta'], self.pars['zeta'], self.pars['d_max'], self.pars['kappa']])

        self.C = C_D[0].T
        self.D = C_D[1].T
        self.F = (softp(self.pars['alpha']) * self.D.T + self.pars['beta']).T
        self.genfunc = theano.function([self.S, In(self.P, value=np.zeros([1, 1]).astype(config.floatX))], self.F, on_unused_input='ignore')


class ML_phys(GenerativeModel):
    """
    Nonlinear model as described in the paper, allows for facilitation and saturation.
    Sigmoid and softplus functions are used to ensure valid parameters. Otherwise training might crash.
    delta_t is the constant bin size, and is not trained. 

    C_(t) = s + exp(δ/τ) * C_(t-1)
    D_(t) = minimum(1/softplus(γ),(D_(t-1) * maximum(0,1 - δ/softplus(kappa)*(1+softplus(γ)*((softplus(c0)+C_(t)) ^ η - softplus(c0) ^ η))) + δ/softplus(kappa)*((softplus(c0)+C_(t)) ^ η - softplus(c0) ^ η)))
    F_(t) = softplus(α) * D_(t) + β + sigmoid(σ)
    """
    def __init__(self, init_params, inf_params=None, X=T.matrix(), S=T.matrix(), P=T.matrix(), batch_size=1, rng=None):
        super().__init__(init_params, inf_params, X, S, P, batch_size, rng)

        self.type = 'ML_phys'
        start = T.zeros([self.S.shape[0]])

        def one_step(s, C_tm1, D_tm1, delta_t, tau, eta, c0, kappa, gamma):
            C = s + T.exp(-delta_t / tau) * C_tm1
            D = T.minimum(1/softp(gamma),(D_tm1 * T.maximum(0,1. - delta_t/softp(kappa)*(1+softp(gamma)*((softp(c0)+C)**eta - softp(c0)**eta))) + delta_t/softp(kappa)*((softp(c0)+C)**eta - softp(c0)**eta)))
            return C, D

        C_D, updates = theano.scan(fn=one_step,
                                   sequences=[self.S.T],
                                   outputs_info=[start, start],
                                   non_sequences=[self.pars['delta_t'], self.pars['tau'], self.pars['eta'], self.pars['c0'], self.pars['kappa'], self.pars['gamma']])

        self.C = C_D[0].T
        self.D = C_D[1].T
        self.F = (softp(self.pars['alpha']) * self.D.T + self.pars['beta']).T
        self.genfunc = theano.function([self.S, In(self.P, value=np.zeros([1, 1]).astype(config.floatX))], self.F, on_unused_input='ignore')