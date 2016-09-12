__author__ = 'gru'

from os import listdir, makedirs
from os.path import isfile, join, exists
import platform

import soundfile as sf
import numpy as np

from core.audio_cwe import watermarking_utils
from core.audio_cwe.watermarking_scheme import HistBasedWMSystem
from core.audio_cwe.xs_wm_scheme import XsWMSystem
from core.audio_cwe.schmitz_wm_scheme import SchmitzWMSystem

# Some predefined watermarking parameters
max_seed = 9999999
min_seed = 1
la = 2.5
bin_threshold = 2
num_bins = 1500
step = 9

# retrieve test files to mark
input_dir = '../../../res/testing/original_test_files/selection'

# create output dir
output_dir = join(input_dir.rsplit('/', maxsplit=2)[0],
                  'transparency_eval/odg_vs_capacity2/'+'/marked_test_files')
makedirs(output_dir, exist_ok=True)

# retrieve files
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith('.wav')]

if platform.system() == 'Darwin' and files[0].startswith("."):
    files = files[1:]  # exclude .ds_store on mac

for i, f in enumerate(files):
    # Increasing the watermark's length
    for l in range(4, 10):
        wmk_len = 2**l
        samples, samplerate = sf.read(input_dir + '/' + f, dtype=np.int16)
        samples = samples[:, 0]  # only mono
        orig_mean = HistBasedWMSystem.mean_of_absolute_values(samples)

        # Choose seed pseudo-randomly
        seed = int((max_seed - min_seed) * np.random.random_sample() + min_seed)

        # Generate random mark
        wmk = []
        for c in range(wmk_len):
            wmk.append(round(np.random.sample()))

        # Construct watermarking system
        wm_sys = XsWMSystem(la=la, num_bins=num_bins, threshold=bin_threshold,
               step=step)
        #SchmitzWMSystem(step=9)
        #XsWMSystem(la=la, num_bins=num_bins, threshold=bin_threshold,
            # step=step)

        marked_samples, bp = wm_sys.embed_watermark(samples, wmk, key=seed)
        sf.write(output_dir + '/' + f[:-4]+'_marked_' + str(len(wmk))+'_bits'
                 + '.wav', marked_samples, samplerate)

        iv = wm_sys.get_params()

        watermarking_utils.dump_params(output_dir, f[:-4]+'_' + str(len(
            wmk))+'_bits', iv=iv, key=bp, mark=wmk)


