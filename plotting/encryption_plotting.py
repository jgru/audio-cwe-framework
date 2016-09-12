__author__ = 'gru'

# Imports some dependencies

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from core.audio_cwe import catmap_encryption
from core.audio_cwe import watermarking_utils
plt.rcParams['agg.path.chunksize'] = 10000
# Specify .wav-file to load
input_file = '../../res/testing/original_test_files/SQAM/64.wav'

# Read audio file
samples, samplerate = sf.read(input_file, dtype=np.int16)
samples = samples[:, 0]

# Set key material
a = 16
b = 53
n = 5

# Encrypt audio file
cipher_samples_1, padding_length = catmap_encryption.encrypt(samples, a, b, 1)
cipher_samples_n, padding_length = catmap_encryption.encrypt(samples, a, b, n)
print('PAdding length:', padding_length)

# Plot the waveform of original cover work
timeArray = np.arange(0, len(samples),
                      1)  # contains sample number (0,1,2,3,...)
timeArray = timeArray / samplerate  # contains time label in seconds

fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax0.plot(timeArray, samples, color='k')
ax0.set_title("Waveform of original signal")
ax0.set_ylabel('Amplitude')
ax0.set_xlabel('Time in s')
ax0.set_ylim([-1 * 2 ** 15, 2 ** 15])
ax0.set_xlim([0, len(cipher_samples_1) / samplerate])

# Generate the sample ids (0,1,2,3,...)
timeArray = np.arange(0, len(cipher_samples_1), 1)
# Generate time labels in seconds
timeArray = timeArray / samplerate

# Plot the waveform of the encrypted work after 1 iteration
fig0, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax1.set_ylim([-1 * 2 ** 15, 2 ** 15])
ax1.plot(timeArray, cipher_samples_1, color='k')
ax1.set_title("Waveform of signal after 1 iteration")
ax1.set_ylabel('Amplitude')
ax1.set_xlabel('Time in s')
ax1.set_xlim([0, len(cipher_samples_1) / samplerate])

timeArray = np.arange(0, len(cipher_samples_n), 1)
timeArray = timeArray / samplerate

# Plot the waveform of the encrypted work after n iterations
fig0, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax2.plot(timeArray, cipher_samples_n, color='k')
ax2.set_title("Waveform of encrypted signal after " + str(n)
              + " iterations")
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Time in s')
ax2.set_ylim([-1 * 2 ** 15, 2 ** 15])
ax2.set_xlim([0, len(cipher_samples_n) / samplerate])

plt.show()

psnr_1 = watermarking_utils.psnr(samples, cipher_samples_1[:len(
    cipher_samples_1) - padding_length])
print("PSNR after 1 iteration:\n", psnr_1)

psnr_n = watermarking_utils.psnr(samples, cipher_samples_n[
    len(cipher_samples_n) - padding_length])
print("PSNR after ", n, " iterations:\n", psnr_n)