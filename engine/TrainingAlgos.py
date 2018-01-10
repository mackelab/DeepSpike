"""
Created on Fri Apr  1 13:04:14 2016

@author: artur
"""

from RecognitionModel import *
import pickle

import time
import sys

from data_funcs import *
from perf_funcs import *


class TrainingAlgo(object):
    """
    Class that controls the training process. 
    """
    def __init__(self, rec_params,  rec_model, batch_size, n_samples, filename, rng, use_patience):
        """
        :param rec_params: Dictionary with recurrent model parameters.
        :param rec_model: Rec. model class, i.e. GRU_BernoulliRecognition for non-factorizing posterior or BernoulliRecognition of factorizing posterior.
        :param batch_size: Batch size
        :param n_samples: Number of samples, only relevant when using VIMCO
        :param filename: String for the name of the file that stores the model throughout training, if None nothing is saved.
        :param rng: RandomNumber generator shared across the whole model to guarantee reproducability
        :param use_patience: Whether to use early stopping with patience.
        """

        self.X = T.matrix('X')  # symbolic variable for the data
        self.S = T.matrix('S')  # symbolic variable for samples
        self.P = T.matrix('P')  # symbolic variable for inferred gen. params
        self.Z = T.matrix('Z')  # symbolic variable for true spikes
        self.BL = T.matrix('BL')  # symbolic variable for the base line

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.buffers = [1, 1]
        self.lr_decay = 0.9995
        self.gen_lr_fac = 20
        self.superres = 1
        self.resample = 60

        self._filename = filename
        self._use_patience = use_patience
        self.got_gen = False
        self.conc_cache = False

        self.exp_params = None
        self.description = None

        self.col_dict = {}

        if self._use_patience:
            self._patience = 100000
            self._patience_increase = 2

        self.mrec = rec_model(rec_params, self.X, self.S, self.P, self.rng, self.batch_size)
        if 'superres' in rec_params.keys(): self.superres = rec_params['superres']; self.mrec.superres = rec_params['superres']

        self.facs = [1]
        self._iter_count = 0

        self.validation_score = 'corr_0'
        self.val_sign = 1
        self.best_validation_score = -np.inf

    def save_object(self, filename):
        """
        Saves model to file filename
        :param filename: Filename
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def update_patience(self, validation_score):
        """
        Update patience parameter
        :param validation_score: Metric that needs to increase in order to prolong training
        """
        if self.val_sign*validation_score > self.best_validation_score:
            self._patience = max(self._patience, self._iter_count * self._patience_increase)
            self.best_validation_score = self.val_sign*validation_score
            if self._filename:
                self.save_object(self._filename + '.pkl')

    def check_stuck(self, epoch):
        """
        Aborts training if the network got stuck as determined by the number of predicted spikes not changing
        :param epoch: training epoch
        """
        if epoch > 5 and np.abs(list(self.col_dict['factor_0'].values())[-1] - list(self.col_dict['factor_0'].values())[-2]) < 1e-5 and np.abs(
                        list(self.col_dict['factor_0'].values())[-2] - list(self.col_dict['factor_0'].values())[-3]) < 1e-5:
            print('Training Stuck')
            return True
        else:
            return False

    def fit(self, trainf, trains, max_epochs=100, learning_rate=1e-3, print_output=True, print_freq=1):
        """
        Function carrying out the training.
        :param trainf: List of fluorescence traces for the training. 
        :param trains: List of spikes (ground truth) for supervised training. Empty list for unsupervised learning.
        :param max_epochs: Epochs, here not the number of passes through the whole datasets, but number of printouts.
        :param learning_rate: Learning rate
        :param print_output: Whether to continously print training progress.
        :param print_freq: Iterations between evaluating the training progress.
        """
        epoch = 0
        self.init_dicts()
        self.print_freq = print_freq

        self.lr = theano.shared(np.array(learning_rate, dtype=theano.config.floatX), name='lr')
        self.lr_decay = np.array(self.lr_decay, dtype=theano.config.floatX)

        if self.conc_cache: self.gen_inits = [[[0, 0, 0] for _ in range(len(trainf[i])+1)] for i in range(len(trainf))]
        self.param_updaters = self.update_params()

        ''' TRAINING '''
        last_print = 0
        tot_t = 0
        while epoch < max_epochs and (self._patience > self._iter_count):

            if self.check_stuck(epoch): break

            t0 = time.time()
            self.lr.set_value(self.lr.get_value() * self.lr_decay)
            
            Iterator = self.DataIterator(self, trainf, trains)

            batches, cells, z_true, indices = zip(*Iterator)
            tot_cost = 0

            for x, c, z, inds in zip(batches, cells, z_true, indices):

                self._iter_count += 1
                ret_dict = self.train(x, c, z, inds, self.param_updaters)
                tot_cost += ret_dict['cost']

            tot_t += (time.time()-t0)
            if (self._iter_count - last_print) < self.print_freq: continue
            self.last_c = (-1, -1)

            self.col_dict['cost_hist'][self._iter_count] = tot_cost / (self._iter_count-last_print)
            updatetime = 1000 * (tot_t) / (self._iter_count-last_print)
            last_print = self._iter_count
            tot_t = 0

            ''' EVALUATION '''

            t0 = time.time()

#             if hasattr(self,'plot_set'):
#                 plot_preds_bl(self,self.plot_set[0],self.plot_set[1],trace=self.plot_set[2],figsize=(35,7),ts=[0,20000])
#                 plt.show()

            for eval_set,i in zip(self.eval_sets, range(len(self.eval_sets))):

                self.eval_func(eval_set, i)

            if self._use_patience: self.update_patience(self.col_dict[self.validation_score][self._iter_count])
                    
            evaltime = time.time() - t0

            self.col_dict['update_time'][self._iter_count] = updatetime
 
            if print_output:

                if print_output:

                    if len(self.eval_sets) == 1:
                        print('{}{:0.3f}'.format('Corr. Train: ', float(self.col_dict['corr_0'][self._iter_count])), end='')
                    else:
                        print('{}{:0.3f}{}{:0.3f}'.format('Corr. Train/Val: ', float(self.col_dict['corr_0'][self._iter_count]), '/', float(self.col_dict['corr_1'][self._iter_count])), end='')
                    if self.got_gen:
                        print('{}{:0.3f}'.format(' Synchro: ', float(self.col_dict['synchro_0'][self._iter_count])), end='')
                        print('{}{}{:0.3f}{}{:0.3f}'.format(' || ', 'MSE Pred./Truth: ', self.col_dict['mse_rec_pred_0'][self._iter_count], ' ',self.col_dict['mse_rec_truth_0'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.col_dict['cost_hist'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Factor: ', self.col_dict['factor_0'][self._iter_count]), end='')
                    print('{}{}{:0.1f}{}{:0.1f}{}'.format(' || ', 'Time upd./Eval.: ', float(updatetime), ' ms ', float(evaltime), ' s'), end='')
                    print('{}{}{}'.format(' || ', 'BatchNr.: ', self._iter_count))

                    sys.stdout.flush()

                sys.stdout.flush()

            if self._filename:

                self.col_dict['description'] = self.description
                self.save_object(self._filename + '_curr.pkl')
                with open(self._filename + '_dicts.pkl', 'wb') as f:
                    pickle.dump(self.col_dict, f)

            epoch += 1
