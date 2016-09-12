__author__ = 'gru'

from os import path

import soundfile as sf
import numpy as np

from core.audio_cwe.xs_wm_scheme import XsWMSystem
from core.audio_cwe import watermarking_utils
from experimental_testing.robustness_eval import tsm_utils


# Define file names
input_file = '../../res/testing/original_test_files/SQAM/58.wav'
output_dir = '../../res/demo'
filename = input_file.split('/')[-1].split('.')[0]
suffix = '_24_bits_xs'

# Load sound file
samples, fs = sf.read(input_file, dtype=np.int16)

# Construct the watermarking system
wm_sys_xs = XsWMSystem(num_bins=224, la=2.5, threshold=50, delta=0.05, step=5)

# Form the watermark w, each list element is embedded in one channel
w = watermarking_utils.construct_watermark(['HdM', 'Stg'])

# Mark the samples
marked_samples, key = wm_sys_xs.embed_watermark(samples, w, key=[1234, 5678])

# Write result to disk
out_file = path.join(output_dir, filename + suffix + '.wav')
sf.write(out_file, marked_samples, fs)

# Write the IV to disk for later detection
params = wm_sys_xs.get_params()
watermarking_utils.dump_params(output_dir, filename + suffix, iv=params,
                               mark=w, key=key)

# Perform a TSM-attack for demo purposes
samples, fs = sf.read(out_file, dtype=np.int16)
tsm_samples = tsm_utils.resample(samples, .70)
out_file = path.join(output_dir, filename + suffix + '_resampled.wav')
sf.write(out_file, tsm_samples, fs)
print(out_file)

out_file = path.join(output_dir, filename + suffix + '_resampled.wav')

# Read the marked, attacked samples
samples, fs = sf.read(out_file, dtype=np.int16)

# Read IV and construct the detector
iv_file = path.join(output_dir, filename + suffix + '_iv')
wm_sys_xs = XsWMSystem.from_file(iv_file)

# Read the mark to form the syn code
mark_file = iv_file[:-2] + 'mark'
mark = np.loadtxt(mark_file, dtype=np.int)
syn = mark #mark[:, :8]

# Read the key material
key_suffix = 'key' + '_(' + str(len(mark)) + ', ' + str(len(mark[0])) + ', 2)'
key_file = iv_file[:-2] + key_suffix
key = watermarking_utils.read_keyfile(key_file)

# Extract the mark
w_2 = wm_sys_xs.extract_watermark(samples, key=key, syn=syn)

# Compare original and recovered mark
watermarking_utils.compare_watermarks(w, w_2)
