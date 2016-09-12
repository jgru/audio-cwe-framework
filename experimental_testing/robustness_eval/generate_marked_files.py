__author__ = 'gru'

from os import listdir, makedirs
from os.path import isfile, join, exists
import platform

import soundfile as sf
import numpy as np

from core.audio_cwe import watermarking_utils
from core.audio_cwe.xs_wm_scheme import XsWMSystem


# some watermarking parameters
max_seed = 100000
min_seed = 30

wmk_len = 56
bin_multiplier = 4
num_bins = wmk_len * bin_multiplier
la = 2.5
bin_threshold = 50
step = 5

# retrieve test files to mark
input_dir = '../../../res/testing/original_test_files/selection'

# create output dir
output_dir = join(input_dir.rsplit('/', maxsplit=2)[0],
                  'robustness_eval/' + str(wmk_len) + '_bits_step_' + str(
                      step) + '_t_' + str(
                      bin_threshold) + '/marked_test_files')

if not exists(output_dir):
    makedirs(output_dir, exist_ok=True)

# retrieve files
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and
         f.endswith('.wav')]

if platform.system() == 'Darwin' and files[0].startswith("."):
    files = files[1:]  # exclude .ds_store on mac

for i, f in enumerate(files):
    samples, samplerate = sf.read(input_dir + '/' + f, dtype=np.int16)
    samples = samples[:, 0]

    # Choose seed pseudo-randomly
    seed = int((max_seed - min_seed) * np.random.random_sample() + min_seed)

    # Generate random mark
    wmk = []
    for c in range(wmk_len):
        wmk.append(round(np.random.sample()))

    # Construct watermarking system
    wm_sys = XsWMSystem(la=la, num_bins=num_bins, threshold=bin_threshold,
                        step=step)

    # Embed mark
    marked_samples, bp = wm_sys.embed_watermark(samples, wmk, key=seed)
    # Get all parameters
    iv = wm_sys.get_params()

    # Store results
    sf.write(output_dir + '/' + f[:-4] + '_' + str(
        len(wmk)) + '_bits' + '.wav', marked_samples, samplerate)
    watermarking_utils.dump_params(output_dir,
                                   f[:-4] + '_' + str(len(wmk)) + '_bits',
                                   key=bp, iv=iv, mark=wmk)



