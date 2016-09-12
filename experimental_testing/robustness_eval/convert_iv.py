__author__ = 'gru'

from os import listdir, makedirs
from os.path import isfile, isdir, join, split, exists
import os.path
import platform

import soundfile as sf
import numpy as np

from core.audio_cwe import watermarking_utils
from core.audio_cwe.xs_wm_scheme import XsWMSystem


# retrieve marked test files
input_dir = '../../../res/testing/transparency_eval/56_bits_step_5_t_50/odg' \
            '/wave/test'
output_dir = '../../../res/testing/transparency_eval/56_bits_step_5_t_50/odg' \
             '/wave/test'

# retrieve files
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and
         f.endswith('_iv')]
for i, f in enumerate(files):

    iv = np.loadtxt(input_dir+"/"+files[i], dtype=np.float64)
    wm_sys = XsWMSystem(la=iv[1], num_bins=iv[0], threshold=iv[2],
                        orig_mean=iv[-1], step=5, delta=0.05)
    new_iv = wm_sys.get_params()
    watermarking_utils.dump_params(output_dir, f[:-3], iv=new_iv)