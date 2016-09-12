__author__ = 'gru'


import subprocess as sp


from os import listdir, makedirs
from os.path import *
import platform
import os
import soundfile as sf
import numpy as np

FFMPEG_BIN = "ffmpeg" # prerequisite: ffmpeg has to be installed on the system

# define input directory
input_dir = '../../../res/experimental_testing/robustness_eval/56_bit_mark_step_9_t_65/marked_test_files'
# create output directory structure
output_dir = '../../../res/experimental_testing/robustness_eval/56_bit_mark_step_9_t_65/mp3_compression/attacked_test_files'
os.makedirs(output_dir,exist_ok=True)

# retrieve files
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
wav_files = [f for f in files if f.endswith('.wav')]
avg_bitrate = [245, 225, 190, 175, 165, 130, 115, 100, 85, 65]  # (see: https://trac.ffmpeg.org/wiki/Encode/MP3)

for i, f in enumerate(wav_files):
    out_dir = join(output_dir, f[:-4])
    os.makedirs(out_dir, exist_ok=True)
    for q in np.arange(0,10,2):
        command = [ FFMPEG_BIN, '-y',
            '-i', abspath(join(input_dir, f)),
            '-codec:a', 'libmp3lame',
            '-q:a', str(q),  # control quality (see: https://trac.ffmpeg.org/wiki/Encode/MP3)
            '-ac', '1', # stereo (set to '1' for mono)
            abspath(join(out_dir, f[:-4] + '_' + str(avg_bitrate[q]) + '.mp3'))]
        print(command)
        sp.call(command)
