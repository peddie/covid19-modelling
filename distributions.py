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

def plot_lognormals(params):
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 5, 1000)
    for (mu, sigma) in params:
        plot_lognormal(x, mu, sigma, ax)
    ax.legend()
