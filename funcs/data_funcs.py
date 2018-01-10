import numpy as np
from scipy import signal
from utils import *
import copy
from theano import config


def data_resamp(traces, spikes, fps, spikefps, resample):
    """
    Resamples a dataset of traces and spikes to target frequency using the scipy.signal routine
    :param traces: List of traces belonging to different cells
    :param spikes: List of corresponding spikes
    :param fps: List of fps at which the data was recorded
    :param spikefps: Frequency of the binned spiketrains
    :param resample: Target frequency. Has to be a divider of the spikefps
    """
    traces_test = copy.deepcopy(traces)
    spikes_test = copy.deepcopy(spikes)

    for i in range(len(traces)):
        for j in range(len(traces[i])):
            traces_test[i][j] = signal.resample(traces_test[i][j], int(len(traces_test[i][j])*resample/fps[i])).astype(config.floatX)
            spikes_test[i][j] = rebin(spikes_test[i][j], int(spikefps/resample))[0].astype(config.floatX)
            traces_test[i][j] = traces_test[i][j][:len(spikes_test[i][j])]
            spikes_test[i][j] = spikes_test[i][j][:len(traces_test[i][j])]
    return traces_test, spikes_test


def data_chop(timebins, traces, spikes, sv_inds=[], fps=None, spikefps=60, resample=None, fb=True):
    """
    Resamples data and chops it up for batch training.
    :param timebins: Length of the snippets
    :param traces: List of traces belonging to different cells
    :param spikes: List of corresponding spikes
    :param sv_inds: indices of cells for which to provide the ground truth spikes. For unsupervised learning just provide an empty list. 
    :param fps: List of fps at which the data was recorded
    :param spikefps: Frequency of the binned spiketrains
    :param resample: Target frequency. Has to be a divider of the spikefps
    :param fb: Whether to pass through the data forwards and backwards. This helps using all the available data during training (otherwise buffer regions are discarded)
    """
    n_cells = len(traces)

    if sv_inds is None: sv_inds = list(np.arange(n_cells))

    traces_train = [[] for _ in range(n_cells)]
    spikes_train = [[] for _ in range(n_cells)]

    for i in range(n_cells):

        spikebool = isinstance(spikes, (np.ndarray, list)) and i in sv_inds

        for j in range(len(traces[i])):

            if resample:
                length = int(len(traces[i][j]) * resample / fps[i])
                trace = signal.resample(traces[i][j], length)
                if spikebool: spike = (rebin(spikes[i][j], int(spikefps / resample))).flatten()
            else:
                trace = traces[i][j]
                if spikebool: spike = spikes[i][j]

            trace = trace.astype(config.floatX)
            p = len(trace) // timebins
            if spikebool:
                if len(trace) > len(spike):
                    p = len(spike) // timebins

            trace_f = trace[:p * timebins].reshape([-1, timebins])
            trace_b = trace[-p * timebins:].reshape([-1, timebins])
            if fb: trace_f = np.concatenate((trace_f, trace_b), axis=0)
            traces_train[i].append(trace_f)

            if spikebool:

                spike_f = spike[:p * timebins].reshape([-1, timebins])
                spike_b = spike[-p * timebins:].reshape([-1, timebins])
                if fb: spike_f = np.concatenate((spike_f, spike_b), axis=0)
                spikes_train[i].append(spike_f)

        traces_train[i] = np.vstack(traces_train[i]).astype(config.floatX)
        if spikebool: spikes_train[i] = np.vstack(spikes_train[i]).astype(config.floatX)

    return traces_train, spikes_train


def DatasetMiniBatchIterator(model, traces, spikes):
    """ Basic mini-batch iterator """

    probs = np.array([len(f) for f in traces])
    probs = probs / np.sum(probs)

    if traces is not None:

        n_cells = len(traces)
        us_ind = []

        for i in range(n_cells):
            us_ind.append(list(np.arange(len(traces[i]))))

        rem = list(np.arange(n_cells))

        for _ in range(model.print_freq):

            c = model.rng.choice(rem, 1, p=probs)[0]
            choice = model.rng.choice(np.array(us_ind[c]), model.batch_size, replace=False)

            truth = np.array([0], ndmin=2, dtype=config.floatX)

            if len(spikes[c]) > 0:
                truth = spikes[c][choice]

            yield traces[c][choice], c, truth, choice


def sampleXY(model, N=None, timebins=None, params=None, firing_rate=None, spikes=None, multspikes=1, perfeval=True, rng=None):
    """
    Quickly evaluate generative model reconstructions for given spike trains. Used to calculate the optimal reconstruction.
    """
    pd = copy.deepcopy(params)
    if rng is None: rng = np.random.RandomState()

    if spikes is not None:

        if perfeval: pd['sigma'] = 0.
        N = spikes.shape[0]
        timebins = spikes.shape[1]
        sim_spikes = spikes

    else:
        sim_spikes = np.zeros((N, timebins))
        for i in range(multspikes):
            sim_spikes += rng.binomial(1, firing_rate / multspikes, (N, timebins))

    if model == 'FOOPSI':

        sim_F = np.zeros([N, timebins])

        for t in range(1, timebins):
            sim_F[:, t] = sigmoid(pd['gamma']) * sim_F[:, t - 1] + sim_spikes[:, t]
        sim_F = softp(pd['alpha']) * sim_F + pd['beta'] + pd['sigma'] * rng.normal(size=timebins)

        return [sim_F.astype(config.floatX), sim_spikes.astype(config.floatX)]

    if model == 'SCDF':

        sim_C = np.zeros([N, timebins])
        sim_D = np.zeros([N, timebins])
        sim_F = np.zeros([N, timebins])

        for t in range(1, timebins):
            sim_C[:, t] = sigmoid(pd['gamma']) * sim_C[:, t - 1] + sim_spikes[:, t]
        for t in range(1, timebins):
            sim_D[:, t] = softp(pd['eta']) * (sim_C[:, t] ** (1 + softp(pd['zeta']))) * \
                          (pd['d_max'] - np.minimum(pd['d_max'], sim_D[:, t - 1])) + sigmoid(pd['kappa']) * np.minimum(pd['d_max'], sim_D[:, t - 1])
            sim_F[:, t] = softp(pd['alpha']) * sim_D[:, t] + pd['beta'] + pd['sigma'] * rng.normal()

        return [sim_F.astype(config.floatX), sim_spikes.astype(config.floatX)]

    if model == 'ML_phys':

        sim_C = np.zeros([N, timebins])
        sim_D = np.zeros([N, timebins])
        sim_F = np.zeros([N, timebins])

        for t in range(1, timebins):
            sim_C[:, t] = np.exp(-pd['delta_t'] / pd['tau']) * sim_C[:, t - 1] + sim_spikes[:, t]
        for t in range(1, timebins):
            sim_D[:, t] = np.minimum(1 / softp(pd['gamma']), (sim_D[:, t - 1] * np.maximum(0, 1. - pd['delta_t'] / softp(pd['kappa']) *
                          (1 + softp(pd['gamma']) * ((softp(pd['c0']) + sim_C[:, t]) ** pd['eta'] - softp(pd['c0']) ** pd['eta']))) + pd['delta_t'] / softp(pd['kappa']) *
                          ((softp(pd['c0']) + sim_C[:, t]) ** pd['eta'] - softp(pd['c0']) ** pd['eta'])))

            sim_F[:, t] = softp(pd['alpha']) * sim_D[:, t] + pd['beta'] + pd['sigma'] * np.random.normal()

        return [sim_F.astype(config.floatX), sim_spikes.astype(config.floatX)]
