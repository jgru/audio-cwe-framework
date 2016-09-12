__author__ = 'gru'

from pylab import *
import matplotlib.pyplot as plt

from experimental_testing import latex_table_utils


ber_values_resamp_orig = np.loadtxt(
    '../../res/testing/robustness_eval/32_bits_step_5_t_50/resampling'
    '/BER_resampling_attack_32_bits')
ber_values_timestretch = np.loadtxt(
    '../../res/testing/robustness_eval/32_bits_step_5_t_50/timestretching'
    '/BER_timestretching_attack_32_bits')

# Remove every second value to match dimensions of ber_values_timestretch
ber_values_resamp = ber_values_resamp_orig
ber_values_resamp = np.zeros_like(ber_values_timestretch)

for i in range(len(ber_values_resamp_orig)):
    for j in range(len(ber_values_resamp_orig[i]) - 2):
        if j % 2 == 0:
            ber_values_resamp[i // 2] = ber_values_resamp_orig[i][j]


legend_labels = ["Resample mode", "Pitch-invariant mode"]
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(0, len(ber_values_resamp[0]), 1)
y = np.mean(ber_values_resamp, axis=0)
x2 = np.arange(0, len(ber_values_timestretch[0]), 1)
y2 = np.mean(ber_values_timestretch, axis=0)
ax.plot(x, y, 'bo-', label=legend_labels[0])
ax.plot(x2, y2, 'ro-', label=legend_labels[1])

ax.set_title("Robustness against TSM")
ax.set_ylabel('Average BER')
ax.set_xlabel('Timestretching in %')
ax.set_ylim(0.0, 0.1)
ticks = ['-30%', '-20%', '-10%', '0%', '+10%'
                                       '', '+20%', '+30%']
ax.set_xticklabels(ticks, rotation=45)

plt.legend(bbox_to_anchor=(0, 0, 1, 1), loc=1, borderaxespad=0.8)
plt.show()

header_row = np.array([
    ["Strength", "Mean", "Tenor singing", "Female speech", "Guitar",
     "Soloists (Verdi)", "Choir (Orff)", "Abba", "Eddie Rabbit"]])
table_data = np.vstack((y2, ber_values_timestretch))
table_data = np.around(table_data, 3)
table_data = np.vstack((ticks, table_data)).T
table_data = np.append(header_row, table_data, axis=0)

latex_table_utils.print_table(table_data)