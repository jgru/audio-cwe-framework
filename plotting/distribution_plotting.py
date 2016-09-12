__author__ = 'gru'

from scipy import stats

# some additional dependencies
from pylab import *
import soundfile as sf


def generate_padding(plain_samples, blocksize):
    padding_length = blocksize * blocksize - len(plain_samples)

    bincount = (-1 * (np.amin(plain_samples)))
    bincount += (np.amax(plain_samples) + 1)  # add the zero bin
    hist, bins = np.histogram(plain_samples, bincount, density=True)
    pk = hist / np.sum(hist)

    xk = np.arange(np.amin(plain_samples), np.amax(plain_samples) + 1)
    sampdist = stats.rv_discrete(name='Sample distribution', values=(xk, pk))
    randsamples = sampdist.rvs(size=padding_length)

    return randsamples


def calc_block_size(plain_samples):
    blocksize = np.sqrt(len(plain_samples))

    if blocksize % 1 > 0:
        blocksize += 1

    blocksize = np.floor(blocksize)

    return blocksize

# load samples, which should be padded with a
samples, samplerate = sf.read('../../res/rock_excerpt_1.wav', dtype=np.int16)
# use only one channel
samples = samples[:, 0]
print(len(samples))



# generate the minimum necessary amount of pseudo random padding
bs = calc_block_size(samples)

padding = generate_padding(samples, bs)
print(len(padding), " padding samples generated")
padding = np.append(samples, padding)

# Plot the histograms of signal and padding
histMin = np.amin(samples)
histMax = np.amax(samples)

hist0, bins0 = np.histogram(samples, 100, (histMin, histMax))
width0 = 0.7 * (bins0[1] - bins0[0])
center0 = (bins0[:-1] + bins0[1:]) / 2

hist1, bins1 = np.histogram(padding, 100, (histMin, histMax))
width1 = 0.7 * (bins1[1] - bins1[0])
center1 = (bins1[:-1] + bins1[1:]) / 2

fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax0.bar(center0, hist0, align='center', width=width0)
ax0.set_title("Histogram of signal")

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax1.bar(center1, hist1, align='center', width=width1)
ax1.set_title("Histogram of padded signal")


# Plot the discrete distributions
bincount = (-1 * (np.amin(samples)))
bincount += (np.amax(samples) + 1)  # add the zero bin
hist, bins = np.histogram(samples, bincount, density=True)
pk = hist / np.sum(hist)
xk = np.arange(np.amin(samples), np.amax(samples) + 1)
sampdist = stats.rv_discrete(name='Sample Distribution', values=(xk, pk))



# fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax0.plot(xk, sampdist.pmf(xk), 'ro', ms=12, mec='r')
ax0.vlines(xk, 0, sampdist.pmf(xk), colors='r', lw=4)
ax0.set_title("Distribution of signal")
ax0.set_ylim(0.0, 0.00025)

hist, bins = np.histogram(padding, bincount, density=True)
pk = hist / np.sum(hist)
xk = np.arange(np.amin(samples), np.amax(samples) + 1)
paddingdist = stats.rv_discrete(name='Padding Distribution', values=(xk, pk))

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax1.plot(xk, paddingdist.pmf(xk), 'ro', ms=12, mec='r')
ax1.vlines(xk, 0, paddingdist.pmf(xk), colors='r', lw=4)
ax1.set_title("Distribution of padded signal")
ax1.set_ylim(0.0, 0.00025)

plt.show()
