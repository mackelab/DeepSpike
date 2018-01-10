
from TrainingAlgos import *
from data_funcs import *
import collections

class SV(TrainingAlgo):
    """
    Fully supervised training using the binary cross-entropy as loss.
    """
    def __init__(self,
                 rec_params,
                 REC_MODEL,
                 batch_size=20,
                 n_samples=1,
                 filename=None,
                 rng=None,
                 use_patience=True):

        super().__init__(rec_params,REC_MODEL, batch_size, n_samples, filename, rng, use_patience)
        self.algo = 'Supervised'
        self.DataIterator = DatasetMiniBatchIterator

    def train(self, x, cell, z, inds, param_updaters):

        ret_dict = param_updaters(x, z)
        return ret_dict

    def init_dicts(self):

        self.col_dict['exp_params'] = self.exp_params
        self.col_dict['cost_hist'] = collections.OrderedDict([])
        self.col_dict['update_time'] = collections.OrderedDict([])

        for i in range(len(self.eval_sets)):
            self.col_dict['corr_'+str(i)] = collections.OrderedDict([])
            self.col_dict['rmse_'+str(i)] = collections.OrderedDict([])
            self.col_dict['factor_'+str(i)] = collections.OrderedDict([])
            self.col_dict['synchro_' + str(i)] = collections.OrderedDict([])

    def eval_func(self, data, ind):

        rec_dict = self.mrec.get_sample(np.vstack(data['traces'])[:, :data['eval_T']], data['eval_rep'])
        pred_prob = np.mean(rec_dict['Probs'].reshape([-1, data['eval_rep'], data['eval_T']]), axis=1)
        pred_sample = rec_dict['Spikes']

        RMSEs, Corrs, Factor = eval_all(pred_prob, np.vstack(data['spikes'])[:, :data['eval_T']], self.buffers, self.facs)

        self.col_dict['rmse_'+str(ind)][self._iter_count] = RMSEs
        self.col_dict['corr_'+str(ind)][self._iter_count] = Corrs
        self.col_dict['factor_' + str(ind)][self._iter_count] = Factor
        synch = Synchro(pred_sample, np.repeat(np.vstack(data['spikes'])[:, :data['eval_T']], data['eval_rep'], axis=0))
        self.col_dict['synchro_' + str(ind)][self._iter_count] = np.mean(synch)

    def update_params(self):

        ret_dict = {}
        LL = T.mean(self.mrec.eval_log_density(self.Z, self.buffers))
        ret_dict['cost'] = LL

        grads = T.grad(-LL, wrt=self.mrec.get_params())
        updates = lasagne.updates.adam(grads, self.mrec.get_params(), self.lr)

        perform_updates_params = theano.function(
            inputs= [self.X, self.Z],
            on_unused_input='ignore',
            updates=updates,
            outputs=ret_dict)

        return perform_updates_params
