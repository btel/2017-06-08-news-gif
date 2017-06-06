#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data = np.load('glm_parameters.npz')
    simulated = data['simulated']
    time = data['time']

    binsz = time[1] - time[0]
    psth_simulated = simulated.mean(0) / binsz * 1000

    # sub-sample PSTH (0.1 ms -> 0.2 ms)

    n_subsample = 2
    psth_subsampled = psth_simulated.reshape((-1, n_subsample)).mean(1)
    time_subsampled = time[::n_subsample]

    plt.plot(time_subsampled, psth_subsampled)
    plt.ylabel('rate, Hz')
    plt.xlim((6, 20))
    plt.savefig('plot_glm_params_psth.svg')
