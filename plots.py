#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np

def plot_dist_time_series_std(axis, days, obs, samples, label):
    sample_mean = np.mean(samples, axis=0)
    sample_std = np.std(samples, axis=0)

    axis.fill_between(days,
                      means - 3 * stddevs, means + 3 * stddevs,
                      label=f'{label} 3-σ', color='#f0e0e0')
    axis.fill_between(days,
                      means - stddevs, means + stddevs,
                      label=f'{label} 1-σ', color='#e0d0d0')
    axis.plot(days, means,
              label=f'{label} mean')
    axis.plot(days, obs,
              label=f'Recorded {label}',
              marker='D', linewidth=0.8)
    plt.setp(axis.get_xticklabels(), visible=False)

    axis.grid(True)
    axis.set_ylabel(f'Number of {label} cases')
    axis.legend()

def plot_dist_time_series(axis, days, obs, samples, label, plot_log=False, subsample=10):
    for sample in samples[0:-1:subsample, :]:
        axis.plot(days,
                  sample,
                  linewidth=0.05, color='blue', alpha=0.4)
    axis.plot(days,
              samples[-1, :],
              linewidth=0.05, color='blue', alpha=0.4,
              label=f'MCMC draws (subsampled by {subsample})')
    axis.plot(days, np.mean(samples, axis=0),
              label=f'{label} mean', color='red')
    axis.plot(days, obs,
              label=f'Recorded {label}',
              color='orange', marker='D', linewidth=0.8)

    axis.grid(True)
    axis.set_ylabel(f'Number of {label} cases')
    if plot_log:
        axis.set_yscale('log')
    axis.legend()

def plot_posterior_time_series(samples, meas, config, plotf=plot_dist_time_series, plot_log=False):
    days = samples.shape[2]

    fig = plt.figure(figsize=(15, 10))
    fig.set_tight_layout(True)
    gs = gridspec.GridSpec(3, 1)
    country = config['country']
    fig.suptitle(f'Coronavirus epidemic in {country} from {meas.index[0]} to {meas.index[-1]}')

    # Pull out the last axis now so that we can link x-axes (this only matters in interactive plots)
    last_axis = plt.subplot(gs[-1])
    labels = ['Confirmed', 'Recovered', 'Dead']
    for axis_idx in range(3):
        not_last = axis_idx != len(labels) - 1
        axis = plt.subplot(gs[axis_idx], sharex=last_axis) if not_last else last_axis
        plotf(axis, pd.to_datetime(meas.index),
              meas.to_numpy()[:, axis_idx],
              samples[:, axis_idx, :],
              labels[axis_idx],
              plot_log=plot_log)
        if not_last:
            plt.setp(axis.get_xticklabels(), visible=False)

    # Last axis is special for label-handling purposes
    last_axis.set_xlabel(f'Date')
    for tick in last_axis.get_xticklabels():
        tick.set_rotation(22)

    plt.tight_layout()
    plt.show()