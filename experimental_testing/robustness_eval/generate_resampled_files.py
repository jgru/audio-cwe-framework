__author__ = 'gru'

from os import listdir
from os.path import isfile, join
import os.path

import soundfile as sf
import numpy as np

from experimental_testing.robustness_eval import tsm_utils


# Define input directory
input_dir = '../../../res/testing/robustness_eval' \
            '/56_bits_step_5_t_50/marked_test_files'

# Create output directory structure
output_dir = '../../../res/testing/robustness_eval' \
             '/56_bits_step_5_t_50/resampling/attacked_test_files'
os.makedirs(output_dir, exist_ok=True)

# Retrieve marked test files
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
wav_files = [f for f in files if f.endswith('.wav')]

# Set resampling factors
factors = np.arange(0.70, 1.30, 0.1)

# Perform resampling for all files in input_dir and store results in output_dir
for i, f in enumerate(wav_files):
    samples, fs = sf.read(join(input_dir, f), dtype=np.int16)
    file_dir = join(output_dir, f[:-4])
    os.makedirs(file_dir, exist_ok=True)
    for j, factor in enumerate(factors):
        resampled_samples = tsm_utils.resample(samples, factor)
        sf.write(join(file_dir, f[:-4] + '_resample_' + str(factor) + '.wav'), resampled_samples, fs)
