__author__ = 'gru'

from os import listdir, path
from os.path import isfile, join

import soundfile as sf
import numpy as np

from core.audio_cwe import watermarking_utils


# retrieve test files
input_dir_a = '../../../res/testing/transparency_eval/fidelity_vs_capacity' \
              '/wave/ref'
input_dir_b = '../../../res/testing/transparency_eval/fidelity_vs_capacity' \
              '/wave/test'

# create output dir
output_dir = input_dir_b.rsplit('/', maxsplit=2)[0]

# retrieve ref files and test files
ref_files = [f for f in listdir(input_dir_a) if isfile(join(input_dir_a, f))]
ref_files = [f for f in ref_files if f.endswith('.wav')]

files = [f for f in listdir(input_dir_b) if isfile(join(input_dir_b, f))]
test_files = [f for f in files if f.endswith('.wav')]

test_files = sorted(test_files, key=lambda x: int(x.split('_')[2]))
test_files = sorted(test_files, key=lambda x: int(x.split('_')[0]))
test_files = np.array(test_files).reshape(-1, len(test_files)//len(ref_files))

snr_values = np.zeros_like(test_files, dtype=np.float)
for i, rf in enumerate(ref_files):
    print(rf)
    ref_samples, samplerate = sf.read(path.join(input_dir_a, rf),
                                      dtype=np.int16)
    for j, tf in enumerate(test_files[i]):
        print(tf)
        test_samples, samplerate = sf.read(path.join(input_dir_b, tf),
                                           dtype=np.int16)
        snr_values[i][j] = watermarking_utils.snr(ref_samples[:, 0],
                                                  test_samples[:, 0])

        print(i, '-', j, '--', snr_values[i][j])

# save result
np.savetxt(path.join(output_dir, 'snr_over_capacity'), snr_values, fmt='%s')

