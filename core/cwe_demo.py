__author__ = 'gru'

from os import path

import soundfile as sf
import numpy as np

from core.audio_cwe.xs_wm_scheme import XsWMSystem
from core.audio_cwe import catmap_encryption
from core.audio_cwe import watermarking_utils


input_file = '../../res/SQAM/64.wav'
out_dir = '../../res/demo'

# Read audio file and encrypt it
samples, samplerate = sf.read(input_file, dtype=np.int16)
# Key material
a = 1
b = 2
n = 5

samples, padding_length = catmap_encryption.encrypt(samples[:, 0], a, b, n)
output_file = path.join(out_dir, input_file[0:len(input_file) - 4] + '_encrypted.wav')
sf.write(output_file, samples, samplerate)

# Reads encrypted audio file and marks it
samples, samplerate = sf.read(output_file, dtype=np.int16)

# Form mark and syn code
w = watermarking_utils.construct_watermark('CWE')
syn = w[:4]

# Construct watermarking system
wm_system = XsWMSystem()  # init with default values

# Mark the samples
marked_samples, bin_pairs = wm_system.embed_watermark(samples, w, key=1234)
output_file = output_file[0:len(input_file) - 4] + '_marked.wav'
sf.write(output_file, marked_samples, samplerate)

# Read marked and encrypted audio file and decrypt it
samples, samplerate = sf.read(output_file, dtype=np.int16)
samples = catmap_encryption.decrypt(samples, padding_length, a, b, n)
output_file = path.join(out_dir, input_file[0:len(input_file) - 4] + '_decrypted' + '_marked.wav')
sf.write(output_file, samples, samplerate)

# Read decrypted and marked audio file and verify the mark
samples, samplerate = sf.read(output_file, dtype=np.int16)
delta = 0.0  # 5% up and down searching boundaries
w_2 = wm_system.extract_watermark(samples, key=bin_pairs, syn=syn, delta=delta)

# Check, whether the detected watermark is correct
print('=============================================')
print('Result:')
print('---------------------------------------------')
if np.array_equal(w, w_2):
    print('Original watermark and detected watermark match perfectly')
else:
    print('Original watermark and detected watermark do not match ')

print('---------------------------------------------')
print(w)
print(w_2)
