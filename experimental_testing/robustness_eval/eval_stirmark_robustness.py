__author__ = 'gru'

from os import listdir
from os.path import isfile, isdir, join
import os.path
import platform

import soundfile as sf
import numpy as np

from core.audio_cwe import watermarking_utils
from core.audio_cwe.xs_wm_scheme import XsWMSystem


# Specify input directories
input_dir = '../../../res/testing/robustness_eval/32_bits_step_5_t_50' \
            '/marked_test_files'
input_dir_b = '../../../res/testing/robustness_eval/32_bits_step_5_t_50' \
              '/stirmark/attacked_test_files'

# Retrieve marked test files
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

if platform.system() == 'Darwin' and files[0].startswith("."):
    files = files[1:]  # exclude .ds_store on mac

# Retrieve directories of attacked test files
attacked_filedirs = [join(input_dir_b, d) for d in listdir(input_dir_b) if isdir(join(input_dir_b, d))]
print(attacked_filedirs)

wav_files = [f for f in files if f.endswith('.wav')]
iv_files = [f for f in files if f.endswith('iv')]
keys = [f for f in files if f.endswith('key')]
marks = [f for f in files if f.endswith('mark')]

results = []
for i, f in enumerate(wav_files):
    print(f)

    # Init WM system
    wm_sys = XsWMSystem.from_file(join(input_dir, iv_files[i]))
    # Read key
    bin_pairs = watermarking_utils.read_keyfile(input_dir + "/" + keys[i])
    # Read embedded WMK
    wmk = np.loadtxt(input_dir + "/" + marks[i], dtype=np.int)

    attacked_files = [f for f in listdir(attacked_filedirs[i]) if
                      isfile(join(attacked_filedirs[i], f)) and f.endswith('.wav')]

    for j, af in enumerate(attacked_files):
        print(af)

        # Read marked and attacked sound file
        samples, samplerate = sf.read(join(attacked_filedirs[i], af), dtype=np.int16)

        # Extract watermark
        recovered_wmk = wm_sys.extract_watermark(samples, syn=wmk, key=bin_pairs)

        # Check, whether the detected watermark is correct
        print('=============================================')
        print('Result:')
        print('---------------------------------------------')
        if np.array_equal(wmk, recovered_wmk):
            print('Original watermark and detected watermark match perfectly')
        else:
            print('Original watermark and detected watermark do not match ')
        while len(wmk) > len(recovered_wmk):
            recovered_wmk = np. append(recovered_wmk, 0)

        ber = watermarking_utils.calc_bit_error_rate(wmk, recovered_wmk)
        print('BER: ', ber)
        results.append((af, ber))
        print('---------------------------------------------')
        print(wmk.tolist())
        print(recovered_wmk)

np.savetxt(join(os.path.split(input_dir_b)[0], 'v2_BER_stirmark_attacks_' +
                str(len(wmk)) + '_bits'), results, fmt='%s')
