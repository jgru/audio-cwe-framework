__author__ = 'gru'

import numpy as np
from pylab import*
import matplotlib.pyplot as plt


snr_values = np.loadtxt(
    '../../res/testing/transparency_eval/fidelity_vs_capacity'
    '/snr_over_capacity')
print(snr_values)

legend_labels = ["Tenor singing", "Female speech", "Guitar",
                 "Soloists (Verdi)", "Choir (Orff)", "Abba", "Eddie Rabbit"]

fig, ax = plt.subplots(figsize=(12, 6))

for i, arr in enumerate(snr_values):
    x = np.arange(0, len(snr_values[i]), 1)
    y = snr_values[i]
    ax.plot(y, 'o-', label=legend_labels[i])

    #ax.plot(x,y,'ro')


ax.set_title("Fidelity over increasing capacity")
ax.set_ylabel('SNR in dB')
ax.set_xlabel('Length of mark in bits')
#ax.set_ylim((0, 100))

min_len = 4
max_len = 10
ticks = [str(2**i)+' Bits  ' for i in range(min_len, max_len)]
print(ticks)
ax.set_xticklabels(ticks, rotation=45)

plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.01
           )

plt.show()