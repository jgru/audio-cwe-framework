__author__ = 'gru'

# duplicate the left channel

from os import listdir
from os.path import isfile, join

import soundfile as sf
import numpy as np

'''
Duplicates a single channel
'''

input_dir = '../../../res/testing/transparency_eval' \
            '/fidelity_vs_capacity/wave/test'
# retrieve files
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith('.wav')]


for i, f in enumerate(files):
    print(f)
    samples, samplerate = sf.read(input_dir + '/' + f, dtype=np.int16)
    samples2 = np.vstack((samples, samples))
    samples2 = samples2.T
    sf.write(input_dir + '/' + f, samples2, samplerate=samplerate)