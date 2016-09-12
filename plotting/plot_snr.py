__author__ = 'gru'

import numpy as np
from pylab import*
import matplotlib.pyplot as plt

# make to separate plots to align them in latex -> separate subcaptions are needed
snr_values_32_bits = np.loadtxt(
    '../../res/testing/transparency_eval/32_bits_step_5_t_50'
    '/snr/calculated_snr_values_32_bits')
snr_values_56_bits = np.loadtxt(
    '../../res/testing/transparency_eval/56_bits_step_5_t_50'
    '/snr/calculated_snr_values_56_bits')
x = np.arange(0, len(snr_values_32_bits), 1)
y = snr_values_32_bits
y2 = snr_values_32_bits

fig, ax = plt.subplots(figsize=(12, 8))
#ax.plot(x, y, 'k--')
ax.plot(x, y2, '^r')
ax.set_title("Fidelity of signals with a 32 bit mark in a 128 bin histogram")
ax.set_ylabel('SNR in dB')
ax.set_xlabel('Index of soundfile')

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)
ax.plot(x, fit_fn(x), '-b')

# set ticks and tick labels
ticks = np.arange(0, len(snr_values_32_bits), 1)
ax.set_xticks(ticks, minor=True)
ax.set_ylim((0, 100))


x = np.arange(0, len(snr_values_56_bits), 1)
y = snr_values_56_bits
y2 = snr_values_56_bits

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, y2, '^r')
ax.set_title("Fidelity of signals with a 56 bit mark in a 224 bin histogram")
ax.set_ylabel('SNR in dB')
ax.set_xlabel('Index of soundfile')

fit = np.polyfit(x, y, 1)
fit_fn = np.poly1d(fit)
ax.plot(x, fit_fn(x), '-b')

# set ticks and tick labels
ticks = np.arange(0, len(snr_values_56_bits), 1)
ax.set_xticks(ticks, minor=True)
ax.set_ylim((0, 100))

plt.show()
