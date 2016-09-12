__author__ = 'gru'

import numpy as np
from pylab import*
import matplotlib.pyplot as plt

# make to separate plots to align them in latex -> separate subcaptions are needed
odg_values_32_bits = np.loadtxt(
    '../../res/testing/transparency_eval/32_bits_step_5_t_50/odg/odg_32_bits_l_128')
odg_values_56_bits = np.loadtxt(
    '../../res/testing/transparency_eval/56_bits_step_5_t_50/odg/odg_56_bits_l_224')

print(len(odg_values_32_bits))
print(len(odg_values_56_bits))

x1 = np.arange(0, len(odg_values_32_bits), 1)
y1 = odg_values_32_bits

fig, ax = plt.subplots(figsize=(12, 8))
#ax.plot(x, y, 'k--')

ax.plot(x1, y1, '^r')
ax.set_title("ODG of signals with a 32 bit mark - 128 bins")
ax.set_ylabel('ODG in ITU-R Grades')
ax.set_xlabel('Index of soundfile')

fit = np.polyfit(x1, y1, 1)
fit_fn = np.poly1d(fit)
ax.plot(x1, fit_fn(x1), '-b')

# set ticks and tick labels
ticks = np.arange(0, len(odg_values_32_bits), 1)
ax.set_xticks(ticks, minor=True)
ax.set_xlim((0, len(odg_values_32_bits)))
ax.set_ylim((-4.0, 0.0))


x2 = np.arange(0, len(odg_values_56_bits), 1)
y2 = odg_values_56_bits

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x2, y2, '^r')
ax.set_title("ODG of signals with a 56 bit mark - 224 bins")
ax.set_ylabel('ODG in ITU-R Grades')
ax.set_xlabel('Index of soundfile')

fit = np.polyfit(x2,y2,1)
fit_fn = np.poly1d(fit)
ax.plot(x1, fit_fn(x1), '-b')

# set ticks and tick labels
ticks = np.arange(0, len(odg_values_56_bits), 1)
ax.set_xticks(ticks, minor=True)
ax.set_ylim((-4.0, 0.0))
ax.set_xlim((0, len(odg_values_56_bits)))

plt.show()
