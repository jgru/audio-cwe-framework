__author__ = 'gru'

from os import path

import numpy as np
import soundfile as sf

from core.audio_cwe import watermarking_utils
from core.audio_cwe.xiang_wm_scheme import XiangWMSystem

# Define file names
input_file = '../../res/testing/original_test_files/SQAM/46.wav'
output_dir = '../../res/demo'
filename = input_file.split('/')[-1].split('.')[0]
suffix = '_24_bits_xiang'

# Read audio data
samples, fs = sf.read(input_file, dtype=np.int16)

# Form the watermark w, each list element is embedded in one channel
w = watermarking_utils.construct_watermark(['HdM', 'Stg'])

# Construct the Xiangs watermarking system
key = 2.0
wm_system_xiang = XiangWMSystem(la=key, num_bins=3 * len(w[0]), threshold=1.3,
                                delta=0.01)

# Mark the samples
marked_samples = wm_system_xiang.embed_watermark(samples, w, key=key)

# Store results on disk
out_file = path.join(output_dir, filename + suffix + '.wav')
sf.write(out_file, marked_samples, fs)
watermarking_utils.dump_params(output_dir, filename + suffix,
                               iv=wm_system_xiang.get_params(), mark=w)

# Read marked file
samples, fs = sf.read(out_file, dtype=np.int16)

# Construct detector from stored IV
iv_file = path.join(output_dir, filename + suffix + '_iv')
wm_sys_xiang = XiangWMSystem.from_file(iv_file)

# Read mark and form syn
mark_file = iv_file[:-2] + 'mark'
mark = np.loadtxt(mark_file, dtype=np.int)
syn = mark[:, :8]

w_2 = wm_system_xiang.extract_watermark(samples, syn=syn)

# Check result
watermarking_utils.compare_watermarks(w, w_2)