__author__ = 'gru'

from os import listdir
from os.path import isfile, join

import soundfile as sf
import numpy as np

from core.audio_cwe import watermarking_utils

input_dir_a = '../../../res/testing/original_test_files/SQAM'
input_dir_b = '../../../res/testing/transparency_eval' \
              '/56_bits_step_5_t_50/snr/marked_test_files'

# retrieve files
orig_files = [f for f in listdir(input_dir_a) if isfile(join(input_dir_a, f))]
orig_files = [f for f in orig_files if f.endswith('.wav')]

marked_files = [f for f in listdir(input_dir_b) if
                isfile(join(input_dir_b, f))]
marked_files = [f for f in marked_files if f.endswith('.wav')]

snr_values = []

for i, marked_file in enumerate(marked_files):
    print(orig_files[i])
    print(marked_file)
    print('---------')
    samples_a, samplerate = sf.read(input_dir_a + '/' + orig_files[i],
                                    dtype=np.int16)
    samples_b, samplerate = sf.read(input_dir_b + '/' + marked_file,
                                    dtype=np.int16)
    snr_values.append(watermarking_utils.snr(samples_a[:, 0], samples_b))

# save result
output_dir = input_dir_b.rsplit('/', maxsplit=1)[0]
np.savetxt(join(output_dir, 'calculated_snr_values_56_bits'),
    np.array(snr_values), fmt='%s')

