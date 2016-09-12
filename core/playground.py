__author__ = 'gru'

from core.audio_cwe.watermarking_scheme import HistBasedWMSystem

import soundfile as sf
import numpy as np

# Define file names
input_file = '../../res/testing/original_test_files/SQAM/46.wav'
output_dir = '../../res/demo'
filename = input_file.split('/')[-1].split('.')[0]
suffix = '_24_bits_xs'

# Load sound file
samples, fs = sf.read(input_file, dtype=np.int16)

print(HistBasedWMSystem.mean_of_absolute_values(samples))
print(HistBasedWMSystem.mean_of_absolute_values_v2(samples[:,0]))
print(HistBasedWMSystem.mean_of_absolute_values_v2(samples[:,1]))