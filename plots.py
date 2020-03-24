#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
from functools import partial

def plot_dist_time_series(axis, days, samples, label,
                          obs=None, extra=None,
                          plot_log=False, sample_points_only=False, subsample=10):
    plotf = partial(axis.scatter, alpha=0.1) if sample_points_only else partial(axis.plot, linewidth=0.05, alpha=0.4)
    for sample in samples[0:-2:subsample, :]:
        plotf(days,
              sample,
              color='blue')
    plotf(days,
          samples[-1, :],
          color='blue',
          label=f'MCMC draws (subsampled by {subsample})')
    axis.plot(days, np.mean(samples, axis=0),
              label=f'{label} mean', color='red')
    axis.plot(days, np.median(samples, axis=0),
              label=f'{label} median', color='magenta')
    if obs is not None:
        axis.plot(days, obs,
                  label=f'Recorded {label}',
                  color='orange', marker='D', linewidth=0.8)
    if extra is not None:
        axis.plot(days,
                  np.mean(extra['values'], axis=0),
                  label=extra['label'],
                  color='green')

    axis.grid(True)
    axis.set_ylabel(f'Number of {label} cases')
    if plot_log:
        axis.set_yscale('log')
    axis.legend()

def plot_posterior_time_series(samples, meas, config,
                               plotf=plot_dist_time_series,
                               sample_points_only=False,
                               extra=None,
                               plot_log=False):
    days = samples.shape[2]
    estimates = samples.shape[1]

    fig = plt.figure(figsize=(15, 10))
    fig.set_tight_layout(True)
    gs = gridspec.GridSpec(estimates, 1)
    country = config['country']
    if meas is not None:
        fig.suptitle(f'Coronavirus epidemic in {country} from {meas.index[0]} to {meas.index[-1]}')
    else:
        fig.suptitle(f'Simulated coronavirus epidemic in {country} for {days} days.')

    # Pull out the last axis now so that we can link x-axes (this only matters in interactive plots)
    last_axis = plt.subplot(gs[-1])
    labels = ['Confirmed', 'Recovered', 'Dead', 'Exposed']
    def fmap(f, x):
        if x is None:
            return None
        else:
            return f(x)

    for axis_idx in range(estimates):
        not_last = axis_idx != estimates - 1
        axis = plt.subplot(gs[axis_idx], sharex=last_axis) if not_last else last_axis
        # obs = meas.to_numpy()[:, axis_idx] if not_last else None
        obs = fmap(lambda x: x.to_numpy()[:, axis_idx], meas)
        extra = fmap(lambda x: {
            'values':x['values'][axis_idx, :],
            'label':x['labels'][axis_idx],
        }, extra)

        plotf(axis, pd.to_datetime(meas.index) if meas is not None else range(days),
              samples[:, axis_idx, :],
              labels[axis_idx],
              obs=obs,
              extra=extra,
              sample_points_only=sample_points_only,
              plot_log=plot_log)

        if not_last:
            plt.setp(axis.get_xticklabels(), visible=False)

    # Last axis is special for label-handling purposes
    last_axis.set_xlabel(f'Date')
    for tick in last_axis.get_xticklabels():
        tick.set_rotation(22)

    plt.tight_layout()
    plt.show()