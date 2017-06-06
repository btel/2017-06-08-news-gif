import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rcParams

scale=1.0
fig_width = 5 / scale  # width in inches
fig_height = 2.5 / scale # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
          'axes.labelsize': 8,
          'axes.titlesize': 10,
          'text.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.fontsize':8,
          'font.size': 8,
          'text.usetex': False,
          'figure.figsize': fig_size,
          'font.family': 'sans-serif',
          'lines.linewidth': 0.5}
rcParams.update(params)

from JoyInf import *
from bakerlab import readspt
from patterns import SortSpikes, BinTrains
import numpy as np
import sys

import matplotlib.pyplot as plt

def plot_parameters(glm, bin):
    plt.subplot(131)
    plt.plot(glm.time, np.exp(glm.inputs), 'k-')
    #plt.ylim([-6, 1])
    plt.ylabel('inputs (a.u.)')
    plt.xlabel('time (ms)')
    plt.subplot(132)
    time_from_spike = np.arange(glm.klen)*bin
    plt.plot(time_from_spike, np.exp(glm.kernel), 'k-')
    #plt.ylim([-1, 1])
    plt.ylabel('history (a.u.)')
    plt.xlabel('time after spike (ms)')

rec={'session':'3349a16',
         'electrode':7,
          'cellid':1}
pathstim=fspikestim % rec
pathspt=fspt % rec
sp_win=[5.,20.]
BINSZ=0.1

spt=readspt(pathspt)/200.0
stim=readspt(pathstim)/200.0
trains=SortSpikes(spt,stim,sp_win)
n_trials=len(trains)

time, binned = BinTrains(trains, sp_win, BINSZ)
binned = binned.T[:,:]

kernel_len = int(8/BINSZ)
from spike_glm.glm_numpy import SpikeGLM
glm = SpikeGLM(kernel_len, BINSZ)
glm.train(binned)
glm.time = time
simulated = glm.simulate(binned.shape[0])
plt.figure()
plot_parameters(glm, BINSZ)
plt.subplot(133)
psth_simulated = simulated.sum(0) 
plt.plot(time, psth_simulated)
plt.tight_layout()
np.savez('glm_parameters.npz',
         simulated=simulated,
         time=glm.time,
         history=glm.kernel,
         intensity=glm.inputs)
