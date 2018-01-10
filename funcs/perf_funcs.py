import seaborn as sns
sns.set_style("white")

import numpy as np
import pyspike

from utils import rebin
from data_funcs import sampleXY
import math


def get_pred_mse(model, cell, traces, samples, spikes, p_sample=None):
    """
    Calculates the mse_rec_pred and mse_rec_truth metrics during training.
    """
    buff0 = model.buffers[0]
    buff1 = model.buffers[1]

    size = 1
    for dim in traces[:,buff0:-buff1].shape: size *= dim

    mset = 0

    if p_sample is None:

        model.mgen.load_genparams(cell)
        reconstruction = model.mgen.genfunc(samples)[:,::model.superres]
        reconstruction = reconstruction.reshape(traces.shape[0],-1,traces.shape[-1]).mean(axis=1)

        msep = np.sqrt(np.sum(((traces - reconstruction)**2)[:,buff0:-buff1])/size)

        if spikes is not None:
            mset = np.sqrt(np.sum(((traces - model.mgen.genfunc(spikes)[:,::model.superres])**2)[:,buff0:-buff1])/size)

    else:

        reconstruction = model.mgen.genfunc(samples, p_sample)[:, ::model.superres]
        reconstruction = reconstruction.reshape(traces.shape[0], -1, traces.shape[-1]).mean(axis=1)

        msep = np.sqrt(np.sum(((traces - reconstruction) ** 2)[:, buff0:-buff1]) / size)

        if spikes is not None:

            p_sample = p_sample.reshape(traces.shape[0], -1,p_sample.shape[-1]).mean(axis=1)
            mset = np.sqrt(np.sum(((traces - model.mgen.genfunc(spikes, p_sample)[:, ::model.superres]) ** 2)[:, buff0:-buff1]) / size)

    if model.true_params[cell] is not None:

        if model.base_mse[cell] == 0:
            smoothed_trace, _ = sampleXY(model=model.mgen.type, spikes=spikes, params=model.true_params[cell])
            model.base_mse[cell] = np.mean(np.sqrt(np.sum(((traces - smoothed_trace)**2)[:,buff0:-buff1])/size))

        return np.mean(msep)/model.base_mse[cell], np.mean(mset)/model.base_mse[cell]
    else:
        return np.mean(msep), np.mean(mset)


def Synchro(spikes1, spikes2):
    synchs = []
    for i in range(len(spikes1)):
        sp1 = pyspike.SpikeTrain(np.where(spikes1[i] > 0)[0], len(spikes1[i]))
        sp2 = pyspike.SpikeTrain(np.where(spikes2[i] > 0)[0], len(spikes2[i]))
        synchs.append(pyspike.spike_sync(sp1, sp2))
    return np.array(synchs)


def Corr(pred, truth):
    ccc = []
    for i in range(len(pred)):
        cc = np.corrcoef(pred[i], truth[i])[0, 1]
        if not math.isnan(cc): ccc.append(cc)
    return np.array(ccc)

def Factor(pred, truth):
    facs = []
    for i in range(len(pred)):
        fac = np.sum(pred)/np.sum(truth)
        if not math.isnan(fac): facs.append(fac)
    return np.array(facs)


def RMSE(pred, truth):
    return np.sqrt(np.sum((pred - truth) ** 2, -1) / truth.sum(-1))


def binned_PM(func, facs):
    """Takes a function and a list of factors and returns a new function for perf. measurements with different binning"""
    def wrap_func(pred,truth, facs = facs):

        pms = []
        for f in facs:
            pms.append(func(rebin(pred,f),rebin(truth,f)))
        return pms

    return wrap_func

def eval_performance(model_or_reconstruction, spikes, traces = None, m_func = Corr, buffers = [100,100], facs = None, det = False):
    """Evaluates performance for a model/prediction and a given performance measurement.
    Works on 3d arrays (N_cells, N_traces, N_t)"""

    if traces is None:

        preds = model_or_reconstruction

        if np.ndim(preds) == 3 or isinstance(preds,list):

            preds = np.vstack(preds)
            spikes = np.vstack(spikes)

    else:

        if np.ndim(traces) == 3 or isinstance(traces,list):

            traces = np.vstack(traces)
            spikes = np.vstack(spikes)

        if not det: preds = model_or_reconstruction.mrec.recfunc(traces)
        else:       preds = model_or_reconstruction.mrec.detfunc(traces)

    if facs:
        eval_func = binned_PM(m_func,facs)
    else:
        eval_func = m_func

    return eval_func(preds[:,buffers[0]:-buffers[-1]], spikes[:,buffers[0]:-buffers[-1]])


def eval_all(pred_prob, spikes, buffers, facs, det = True):

    RMSEs = np.ma.masked_invalid(eval_performance(pred_prob, spikes, traces = None, m_func = RMSE, buffers = buffers, facs = facs, det = det)).mean()
    Corrs = np.mean(eval_performance(pred_prob, spikes, traces = None, m_func = Corr, buffers = buffers, facs = facs, det = det))

    if np.ndim(pred_prob) == 3 or isinstance(pred_prob, list):
        pred_prob = np.vstack(pred_prob)
        spikes = np.vstack(spikes)

    Factor = pred_prob[:,buffers[0]:-buffers[-1]].sum(-1)/spikes[:,buffers[0]:-buffers[-1]].sum(-1)
    Factor = np.mean(Factor[Factor<1e8])

    return RMSEs, Corrs, Factor
