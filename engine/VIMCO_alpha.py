from TrainingAlgos import *
from GenerativeModel import *
from data_funcs import *
import collections

class VIMCO(TrainingAlgo):
    """
    Unsupervised training using the VIMCO algorithm to reduce variance in the gradient estimations.
    """
    def __init__(self,
                 init_params,  # dictionary of generative model parameters
                 inf_params,  # list of gen. params that are inferred by the rec. model
                 GEN_MODEL,  # class that inherits from GenerativeModel
                 rec_params,  # dictionary of approximate posterior ("recognition model") parameters
                 REC_MODEL, # class that inherits from RecognitionModel
                 batch_size=20,
                 n_samples=10,
                 n_cells=1,
                 filename=None,
                 rng=None,
                 use_patience=True):

        """
        :param GEN_MODEL: Type of generative model, i.e. FOOPSI, SCDF ...
        """
        self.algo = 'VIMCO'
        super().__init__(rec_params, REC_MODEL, batch_size, n_samples, filename, rng, use_patience)
        self.set_generative(GEN_MODEL, init_params, inf_params, n_cells)
        self.mgen.BL = self.BL
        assert self.mrec.n_genparams == len(self.mgen.inf_params), 'Recognition model returns less parameters then the Generative model expects'
        self.lstsq_bl = False
        self.last_c = (-1, -1)

    def set_generative(self, GEN_MODEL, init_params, inf_params, n_cells):
        """
        Initializes the generative model
        :param GEN_MODEL: Type of generative model, i.e. FOOPSI, SCDF ...
        :param init_params: Inital parameter values, can be either a single dict which initializes all cells with the same parameters of a list of dicts
        :param inf_params: Generative model parameters that are inferred by the recognition model
        :param n_cells: Number of cells
        """
        self.got_gen = True
        self.n_cells = n_cells
        self.gen_model = GEN_MODEL
        self.DataIterator = DatasetMiniBatchIterator

        if isinstance(init_params, dict):
            self.mgen = self.gen_model(init_params, inf_params, self.X, self.S, self.P, self.batch_size, rng=self.rng)
            self.mgen.buffers = self.buffers
            p_dict = {}
            for p in self.mgen.get_params():
                p_dict[p.name] = p.get_value(return_internal_type=True)
            self.mgen.genparam_dicts = [copy.deepcopy(p_dict) for _ in range(n_cells)]
        else:
            assert n_cells == len(init_params), 'Number of cells and number of initial parameter dict does not coincide'
            genparam_dicts = [[] for _ in range(n_cells)]
            for i in range(n_cells):
                self.mgen = self.gen_model(init_params[i], inf_params, self.X, self.S, self.P, self.batch_size, rng=self.rng)
                p_dict = {}
                for p in self.mgen.get_params():
                    p_dict[p.name] = p.get_value(return_internal_type=True)
                genparam_dicts[i] = copy.deepcopy(p_dict)
            self.mgen.genparam_dicts = genparam_dicts

        self.base_mse = [0] * self.n_cells

    def solve_bl(self, residual, lamb=0):
        """
        Performs a regularized least squares regression to estimate the baseline
        :param residual: Residual (Input trace - reconstruction)
        :param lamb: Regularization constant
        """
        A = np.eye(residual.shape[-1])
        diff_op = np.eye(residual.shape[-1]) - np.eye(residual.shape[-1], k=-1)
        return np.linalg.solve(1.1*A + lamb * diff_op, A.dot(residual.T)).T

    def train(self, x, cell, z, inds, param_updaters):
        """
        Trainf function that is called at each iteration
        :param x: Input trace
        :param cell: Cell number
        :param z: Ground truth spikes (only used for supervised training)
        :param inds: Indices of the traces
        :param param_updaters: Updatefunction
        """
        self.mgen.load_genparams(cell)

        rec_dict = self.mrec.get_sample(x, self.n_samples)
        if 'Params' not in rec_dict.keys(): rec_dict['Params'] = np.zeros([1, 1]).astype(config.floatX)
        if 'Baseline' not in rec_dict.keys(): rec_dict['Baseline'] = np.zeros([1, 1]).astype(config.floatX)

        if self.lstsq_bl:
            residual = np.repeat(x, self.n_samples, 0) - self.mgen.genfunc(rec_dict['Spikes'], rec_dict['Params'])
            rec_dict['Baseline'] = self.solve_bl(residual, self.bl_lambda).astype(config.floatX)

        ret_dict = param_updaters(x, z, rec_dict['Spikes'], rec_dict['Params'], rec_dict['Baseline'])

        self.mgen.store_genparams(cell)

        return ret_dict

    def init_dicts(self):
        """
        Initializes dictionary that includes all perfomance metrics that get evaluated throughout training.
        Also created the function that calculates the loglikelihood of ground truth spikes under the current model.
        """
        self.eval_ll_func = theano.function([self.Z, self.X], self.mrec.eval_log_density(self.Z, self.buffers, for_eval=True))
        self.col_dict['exp_params'] = self.exp_params
        self.col_dict['cost_hist'] = collections.OrderedDict([])
        self.col_dict['update_time'] = collections.OrderedDict([])

        for i in range(len(self.eval_sets)):
            self.col_dict['corr_'+str(i)] = collections.OrderedDict([])
            self.col_dict['rmse_'+str(i)] = collections.OrderedDict([])
            self.col_dict['factor_'+str(i)] = collections.OrderedDict([])
            self.col_dict['logl_'+str(i)] = collections.OrderedDict([])
            self.col_dict['marglogl_' + str(i)] = collections.OrderedDict([])
            self.col_dict['synchro_' + str(i)] = collections.OrderedDict([])

            if self.eval_sets[i]['test_gen']:
                self.col_dict['mse_rec_pred_' + str(i)] = collections.OrderedDict([])
                self.col_dict['mse_rec_truth_' + str(i)] = collections.OrderedDict([])

    def eval_func(self, data, ind):
        """
        Calculates various performance metrics, possibly for multiple datasets
        :param data: Data dict including traces and ground truth
        :param ind: Index of the currently evaluated dataset
        """
        rec_dict = self.mrec.get_sample(np.vstack(data['traces'])[:, :data['eval_T']], data['eval_rep'])
        pred_prob = np.mean(rec_dict['Probs'].reshape([-1, data['eval_rep'], data['eval_T']]), axis=1)
        pred_sample = rec_dict['Spikes']

        if 'Params' in rec_dict:
            pred_pars = rec_dict['Params']
        else:
            pred_pars = None
            curr_pars = None

        RMSEs, Corrs, Factor = eval_all(pred_prob, np.vstack(data['spikes'])[:, :data['eval_T']], self.buffers, self.facs)

        self.col_dict['rmse_'+str(ind)][self._iter_count] = RMSEs
        self.col_dict['corr_'+str(ind)][self._iter_count] = Corrs
        self.col_dict['factor_' + str(ind)][self._iter_count] = Factor
        logL = np.squeeze(self.eval_ll_func(np.vstack(data['spikes'])[:, :data['eval_T']], np.vstack(data['traces'])[:, :data['eval_T']]))
        synch = Synchro(pred_sample, np.repeat(np.vstack(data['spikes'])[:, :data['eval_T']], data['eval_rep'], axis=0))
        self.col_dict['logl_' + str(ind)][self._iter_count] = np.mean(logL)
        self.col_dict['synchro_' + str(ind)][self._iter_count] = np.mean(synch)

        if data['test_gen']:

            mse_p, mse_t = [], []
            for c in range(len(data['traces'])):
                curr_sample = pred_sample[:len(data['traces'][c]) * data['eval_rep']]
                pred_sample = pred_sample[len(data['traces'][c]) * data['eval_rep']:]

                if pred_pars is not None:
                    curr_pars = pred_pars[:len(data['traces'][c]) * data['eval_rep']]
                    pred_pars = pred_pars[len(data['traces'][c]) * data['eval_rep']:]

                a, b = get_pred_mse(self, c, data['traces'][c][:, :data['eval_T']], curr_sample, data['spikes'][c][:, :data['eval_T']], p_sample=curr_pars)
                mse_p.append(a)
                mse_t.append(b)

            self.col_dict['mse_rec_pred_' + str(ind)][self._iter_count] = np.array(mse_p).mean()
            self.col_dict['mse_rec_truth_' + str(ind)][self._iter_count] = np.array(mse_t).mean()

    def update_params(self):
        """
        This is the core algorithm performing VIMCO style training. 
        :return: Theano function performing the parameter updates of the generative and recognition model
        """
        ret_dict = {}
        n_samples = self.S.shape[0]//self.batch_size

        I = T.eye(n_samples)
        I = T.repeat(I, self.batch_size, axis=1)
        I = T.reshape(I, (n_samples, n_samples, -1))
        I = I.dimshuffle(2,0,1)
        I = T.cast(I-1, 'int32')  # Turn ones into zeros for nonzero() op.

        log_qz_given_x = self.mrec.eval_log_density(self.S, self.buffers)
        log_px_given_z = self.mgen.eval_log_density(self.buffers, bl=self.lstsq_bl)
        log_pz         = self.mrec.eval_prior(self.buffers)

        """ Dim: (batch_size, n_samples) """

        log_f_i = log_px_given_z+log_pz-log_qz_given_x
        log_f_i_max = T.max(log_f_i, axis=1, keepdims=True)
        log_sum_f = log_f_i_max + T.log(T.sum(T.exp(log_f_i - log_f_i_max), axis = 1, keepdims=True))
        log_omega_i = log_f_i - log_sum_f
        omega_i = T.exp(log_omega_i)

        log_f_j = T.repeat(log_f_i, n_samples, axis=1)
        log_f_j = T.reshape(log_f_j, (self.batch_size, n_samples, n_samples))
        log_f_j = log_f_j[I.nonzero()].reshape([self.batch_size, n_samples - 1, n_samples])
        log_f_j = log_f_j[:, :, ::-1]

        log_f_j_max = T.max(log_f_j, axis=1, keepdims=True)
        log_sum_f_j = log_f_j_max + T.log(T.sum(T.exp(log_f_j - log_f_j_max), axis=1, keepdims=True))
        log_sum_f_j = log_sum_f_j.dimshuffle(0,2,1)
        log_sum_f_j = T.flatten(log_sum_f_j, outdim=2)

        L_K = log_sum_f - T.log(T.cast(n_samples, 'float32'))
        L_i = L_K - log_sum_f_j + T.log(T.cast(n_samples, 'float32')-1)

        log_f_i        = log_f_i.reshape([self.batch_size*n_samples])
        omega_i        = omega_i.reshape([self.batch_size*n_samples])
        log_qz_given_x = log_qz_given_x.reshape([self.batch_size*n_samples])
        L_i            = L_i.reshape([self.batch_size*n_samples])

        total_cost = -T.mean(omega_i*log_f_i + L_i*log_qz_given_x)

        ret_dict['cost'] = -total_cost
        ret_dict['elbo'] = T.reshape(log_f_i, (self.batch_size, n_samples))
        grads_rec = T.grad(total_cost, wrt=self.mrec.get_params(), consider_constant=[omega_i, L_i])

        if self.gradnorm_tot:
            grads_rec = lasagne.updates.total_norm_constraint(grads_rec, self.gradnorm_clip)

        updates = lasagne.updates.adam(grads_rec, self.mrec.get_params(), self.lr)

        if not self.mgen.fixed_gen:
            grads_gen = T.grad(total_cost, wrt=self.mgen.get_params(), consider_constant=[omega_i, L_i])
            self.mgen.gen_updates = lasagne.updates.rmsprop(grads_gen, self.mgen.get_params(), self.gen_lr_fac*self.lr)
            updates.update(self.mgen.gen_updates)

        perform_updates_params = theano.function(
            inputs=[self.X, self.Z, self.S, self.P, self.BL],
            on_unused_input='ignore',
            updates=updates,
            outputs=ret_dict)

        return perform_updates_params