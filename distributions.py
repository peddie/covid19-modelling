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
