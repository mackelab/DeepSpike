from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set_style("white")

from theano import config
import numpy as np
from data_funcs import data_resamp
from utils import *


def plot_od(od, label = None, col = None):
    plt.plot(*zip(*sorted(od.items())), label = label, color = col)


def plot_preds_bl(model, data, cell=0, pred=True, ts=[0, 1000], trace=0, figsize=None, GS=None):

    n_samples = 13

    if figsize is None:
        fig = plt.figure()
    elif isinstance(figsize, tuple):
        fig = plt.figure(figsize=figsize)
    else:
        fig = figsize

    cp = sns.hls_palette(11, l=.4, s=.8)
    try:
        sr = model.superres
    except KeyError:
        sr = 1

    start = 0
    end = - 1

    fluor, spikes = data_resamp(data['traces'], data['spikes'], data['fps'], data['spike_fps'], model.resample)
    fluor = np.array(fluor[cell][trace][ts[0]:ts[1]], ndmin=2).astype(config.floatX)
    spikes = np.array(spikes[cell][trace][sr * ts[0]:sr * ts[1]], ndmin=2).astype(config.floatX)

    model.mgen.load_genparams(cell)
    rec_dict = model.mrec.get_sample(fluor, n_samples)
    pred_prob = rec_dict['Probs']
    pred_spikes = rec_dict['Spikes']
    pred_bl = 0

    if model.mrec.n_genparams:
        pred_pars = rec_dict['Params']

    pred_prob = pred_prob[:, sr * start:sr * end:sr].mean(axis=0)
    if pred:
        if model.mrec.n_genparams:
            pred_fluor = model.mgen.genfunc(pred_spikes, pred_pars)[:, sr * start:sr * end:sr] + pred_bl
            pred_truth = model.mgen.genfunc(np.repeat(spikes, n_samples, 0), pred_pars)[:, sr * start:sr * end:sr]
        else:
            pred_fluor = model.mgen.genfunc(pred_spikes)[:, sr * start:sr * end:sr] + pred_bl
            pred_truth = model.mgen.genfunc(np.repeat(spikes, 1, 0))[:, sr * start:sr * end:sr]

    pred_spikes = pred_spikes[:, sr * start:sr * end:sr]

    s_spikes = np.sum(np.squeeze(pred_prob))
    r_spikes = np.sum(spikes[0, sr * start:sr * end])
    corr = np.corrcoef(rebin(np.squeeze(pred_prob), 3), rebin(spikes[0, sr * start:sr * end], 3))[0, 1]

    pred_inds = [np.where(pred_spikes[i])[0] for i in range(13)]

    if GS is None:
        gs = gridspec.GridSpec(3, 1, height_ratios=([3, 1, 1]))
    else:
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=GS, height_ratios=([2, 1]), hspace=0.)

    axes = []
    for i in range(3):
        axes.append(plt.Subplot(fig, gs[i]))
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        fig.add_subplot(axes[i])

    if GS is None: gs.tight_layout(fig, h_pad=-1.8, rect=(0, 0, 0.7, 1))

    where = np.where(spikes[0, sr * start:sr * end] == 1)[0]
    where2 = np.where(spikes[0, sr * start:sr * end] > 1)[0]

    dt = 1 / model.resample / sr

    c = 1
    t = np.arange(len(fluor[0, start:end])) * dt
    t_sr = np.arange(len(fluor[0, start:end]) * sr) * dt / sr

    for i in where:
        i *= dt
        if c == 1: axes[0].axvline(i, ymin=0., ymax=.1, color='black', label='True Spikes'); c = 0;
        axes[0].axvline(i, ymin=0., ymax=.1, color='black')  # , linestyle='dotted')
        axes[1].axvline(i, color='0.75')
    axes[0].set_xlim([0, t[-1]])
    axes[1].set_xlim([0, t[-1]])
    axes[1].set_ylim([0, 1])

    for i in where2:
        i *= dt
        axes[0].axvline(i, ymin=0., ymax=.2, color='black')
        axes[1].axvline(i, color='0.75')

    for ith, trial in enumerate(pred_inds):
        axes[2].vlines(trial * dt, ith + .5, ith + 1.5, color=cp[0])
    axes[2].set_xlim([0, t[-1]])

    axes[1].set_ylabel('Predicted Probability', fontsize=12)
    axes[1].set_xlabel('Time in Seconds', fontsize=12)
    axes[1].plot(t_sr, pred_prob, color=cp[6])
    axes[0].plot(t, fluor[0, start:end], label="Trace", color=cp[4])
    [axes[0].plot(t, pred_truth[i], label="Prediction | True Spiketrain", linewidth=2) for i in range(len(pred_truth))]
    axes[0].plot(t, pred_fluor.mean(0), label="Prediction | Inferred Spiketrain", color=cp[7], linewidth=3)
    axes[0].legend(loc='upper left', fontsize=12)
    axes[1].text(0.75 * t[-1], 0.85, 'Corr: ' + '{:0.2f}'.format(corr), fontsize=12)
    axes[1].text(0.75 * t[-1], 0.65, 'Spikes: ' + '{:0.2f}'.format(s_spikes) + ' / ' + str(r_spikes), fontsize=12)