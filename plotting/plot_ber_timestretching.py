__author__ = 'gru'

from pylab import *
import matplotlib.pyplot as plt

input_file = '../../res/testing/robustness_eval' \
             '/32_bits_step_5_t_50' \
             '/resampling/BER_resampling_attack_32_bits'
ber_values = np.loadtxt(input_file)
print(ber_values)

legend_labels = ["Tenor singing", "Female speech", "Guitar",
                 "Soloists (Verdi)", "Choir (Orff)", "Abba", "Eddie Rabbit"]
fig, ax = plt.subplots(figsize=(12, 6))

for i, arr in enumerate(ber_values):
    # plot data series
    x = np.arange(0, len(ber_values[i]), 1)
    y = ber_values[i]
    ax.plot(y, 'o-', label=legend_labels[i])
    # print mean
    m = np.mean(y)
    print("{a:3d}: {b:3.6f}".format(a=i, b=m))

    # ax.plot(x,y,'ro')

x = np.arange(0, len(ber_values[0]), 1)
# fit = np.polyfit(x,y,1)
#fit_fn = np.poly1d(fit)
#ax.plot(x, fit_fn(x), 'k--')

ax.set_title("Robustness against Timestretching")
ax.set_ylabel('BER')
ax.set_xlabel('Timestretching in %')
ax.set_ylim(0.0, 1.0)
#ax.set_xlim(0,11)
start = -30
ticks = ['-30%', '-20%', '-10%', '0%', '+10%', '+20%', '+30%']
#ticks = ['-30%',  '-10%',   '+10%',   '+30%']
#print(ticks)
ax.set_xticklabels(ticks, rotation=45)

plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.01)

plt.show()



