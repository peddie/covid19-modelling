#!/usr/bin/env python3

import arviz as az

def get_sample_time_series(fit):
    yhat_indices = tuple([x.startswith('y_hat') for x in fit.column_names])
    yhat_samples = fit.sample[:, :, yhat_indices]
    day_count = yhat_samples.shape[2] // 3
    # This should divide by 3.
    assert(day_count * 3 == yhat_samples.shape[2])
    yhat_samples = yhat_samples.reshape((yhat_samples.shape[0] * yhat_samples.shape[1],
                                         3,
                                         day_count))

    return yhat_samples

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