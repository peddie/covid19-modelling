#!/usr/bin/env python3

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def plot_invgamma(x, alpha, beta, axis):
    axis.plot(x, stats.invgamma.pdf(x, alpha, scale=beta),
              label=f'invgamma({alpha}, {beta})')

def plot_invgammas(params):
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 5, 1000)
    for (alpha, beta) in params:
        plot_invgamma(x, alpha, beta, ax)
    ax.legend()

def plot_lognormal(x, mu, sigma, axis):
    axis.plot(x, stats.lognorm.pdf(x, mu, scale=sigma),
              label=f'lognormal({mu}, {sigma})')
    print(f'e^({mu} + {sigma}^2/2) = {np.exp(mu + sigma**2/2)}')

def plot_lognormals(params, window=5):
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, window, 1000 * window // 5)
    for (mu, sigma) in params:
        plot_lognormal(x, mu, sigma, ax)
    ax.legend()

def plot_gamma(x, a, b, axis):
    q1, q3 = stats.gamma.interval(0.5, a, scale=b)
    median = stats.gamma.median(a, scale=b)
    axis.plot(x, stats.gamma.pdf(x, a, scale=b),
              label=f'gamma({a}, {b}); ({q1:.2f}, {median:.2f}, {q3:.2f})')

def plot_gammas(params, window=5):
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, window, 1000 * window // 5)
    for (a, b) in params:
        plot_gamma(x, a, b, ax)
    ax.legend()

def plot_cauchy(x, a, b, axis):
    axis.plot(x, stats.cauchy.pdf(x, a, scale=b),
              label=f'cauchy({a}, {b})')

def plot_cauchys(params, window=5):
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, window, 1000 * window // 5)
    for (a, b) in params:
        plot_cauchy(x, a, b, ax)
    ax.legend()
