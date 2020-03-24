#!/usr/bin/env python3

import arviz as az
import numpy as np

def get_sample_time_series(fit, n_columns=3):
    yhat_indices = tuple([x.startswith('y_hat') for x in fit.column_names])
    yhat_samples = fit.sample[:, :, yhat_indices]
    day_count = yhat_samples.shape[2] // n_columns
    # This should divide by n_columns.
    assert (day_count * n_columns == yhat_samples.shape[2]), f'Shit\'s fucked: {day_count} * {n_columns} ({day_count * n_columns}) != {yhat_samples.shape[2]}'
    yhat_samples = yhat_samples.reshape((yhat_samples.shape[0] * yhat_samples.shape[1],
                                         n_columns,
                                         day_count))

    return yhat_samples

def compute_sample_log_likelihood(fit, n_columns=3):
    log_likelihood_indices = tuple([x.startswith('log_likelihood') for x in fit.column_names])
    log_likelihood_values = fit.sample[:, :, log_likelihood_indices]
    return np.sum(log_likelihood_values, axis=(1, 2))

def param_posterior_arviz_plots(inferred, variables):
    az.plot_posterior(inferred, var_names=variables, kind='hist')
    az.plot_pair(inferred, var_names=variables, kind='hexbin', colorbar=True, divergences=True)

def run_validate_stan(fit, values):
    fit.diagnose()
    return fit.summary().loc[values, :]

def run_validate_arviz(inferred):
    az.plot_ppc(inferred, data_pairs={'y':'y_hat'})
    loo = az.loo(inferred, pointwise=True)
    az.plot_khat(loo)
    az.plot_loo_pit(inferred, y='y', y_hat='y_hat')
    loo_pit = az.loo_pit(inferred, y='y', y_hat='y_hat')
    bfmi = az.bfmi(inferred)
    if any(bfmi < 0.5):
        print("BFMI warning:", bfmi < 0.5)
    print("LOO analysis:\n", loo)
    return loo, loo_pit, bfmi

def param_validate_arviz(inferred, variables):
    az.plot_mcse(inferred, var_names=variables)
    az.plot_ess(inferred, var_names=variables)
    az.plot_trace(inferred, var_names=variables)

def chain_validate_arviz(inferred, variables):
    for var in variables:
        az.plot_autocorr(inferred, var_names=[var])

def standard_validate_arviz(fit, ll_label, pp_label, observed, variables):
    inferred = az.from_cmdstanpy(fit,
                                 log_likelihood=ll_label,
                                 posterior_predictive=pp_label,
                                 observed_data={'y':observed})
    print("Displaying posterior plots.")
    param_posterior_arviz_plots(inferred, variables)
    print("Validating inference run.")
    _ = run_validate_arviz(inferred)
    print("Validating parameter sampling.")
    param_validate_arviz(inferred, variables)