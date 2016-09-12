__author__ = 'gru'

from os import path

import soundfile as sf
import numpy as np

from core.audio_cwe.schmitz_wm_scheme import SchmitzWMSystem
from core.audio_cwe import watermarking_utils

# Define file names
input_file = '../../res/testing/original_test_files/SQAM/46.wav'
output_dir = '../../res/demo'
filename = input_file.split('/')[-1].split('.')[0]
suffix = '_24_bits_schmitz'

# Load the sound file
samples, fs = sf.read(input_file, dtype=np.int16)

# Construct the watermarking system
wm_system_schmitz = SchmitzWMSystem((-9, 9))
key = [1234, 5678]

# Form the watermark w, each list element is embedded in one channel
w = watermarking_utils.construct_watermark(['HdM', 'Stg'])

# Embed the mark
marked_samples, bin_pairs = wm_system_schmitz.embed_watermark(samples, w,
                                                              key=key)

# Store IV and marked samples on disk
params = wm_system_schmitz.get_params()
watermarking_utils.dump_params(output_dir, filename + suffix, iv=params,
                               mark=w, key=key)

out_file = path.join(output_dir, filename + suffix + '.wav')
sf.write(out_file, marked_samples, fs)


# Read marked samples
samples, fs = sf.read(out_file, dtype=np.int16)

# Construct detector from dumped params
iv_file = path.join(output_dir, filename + suffix + '_iv')
wm_sys_schmitz = SchmitzWMSystem.from_file(iv_file)

key_file = iv_file[:-2] + 'key'
key = np.loadtxt(key_file, np.int)
mark_file = iv_file[:-2] + 'mark'
mark = np.loadtxt(mark_file, dtype=np.int)

w_2 = wm_sys_schmitz.extract_watermark(samples, key=key, length=len(mark[0]))

# Check result
watermarking_utils.compare_watermarks(w, w_2)

