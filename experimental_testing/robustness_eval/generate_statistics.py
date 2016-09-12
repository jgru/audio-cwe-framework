__author__ = 'gru'

from os import listdir
from os.path import isfile, join
import platform

import soundfile as sf
import numpy as np

from core.audio_cwe import watermarking_utils
from core.audio_cwe.watermarking_scheme import HistBasedWMSystem
from experimental_testing import latex_table_utils



# retrieve marked test files
input_dir = '../../../res/testing/robustness_eval/32_bits_step_5_t_50' \
            '/marked_test_files'

ber_values_resample = np.loadtxt(
    '../../../res/testing/robustness_eval/32_bits_step_5_t_50/resampling'
    '/BER_resampling_attack_32_bits')
mean_resamp = np.round(np.mean(ber_values_resample, axis=1), 3)

ber_values_ts = np.loadtxt(
    '../../../res/testing/robustness_eval/32_bits_step_5_t_50/timestretching'
    '/BER_timestretching_attack_32_bits')
mean_ts = np.round(np.mean(ber_values_ts, axis=1), 3)

# retrieve files
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

if platform.system() == 'Darwin' and files[0].startswith("."):
    files = files[1:]  # exclude .ds_store on mac

wav_files = [f for f in files if f.endswith('.wav')]
iv_files = [f for f in files if f.endswith('iv')]
keys = [f for f in files if f.endswith('key')]
marks = [f for f in files if f.endswith('mark')]

labels = ["Tenor singing", "Female speech", "Guitar", "Soloists (Verdi)",
          "Choir (Orff)", "Abba", "Eddie Rabbit"]
results = []
for i, f in enumerate(wav_files):
    results.append(labels[i])

    with open(join(input_dir, iv_files[i]), 'r') as ivf:
        iv = eval(ivf.read())

    bin_pairs = np.loadtxt(join(input_dir, keys[i]))
    wmk = np.loadtxt(join(input_dir, marks[i]), dtype=np.int)

    print(f)
    print(iv_files[i])
    print(keys[i])
    print(marks[i])

    samples, samplerate = sf.read(join(input_dir, f), dtype=np.int16)
    results.append(str(len(samples)))

    hist, bins = HistBasedWMSystem.generate_histogram(samples, iv['la'], int(iv['num_bins']), iv['orig_mean'])
    minbin, maxbin, avgbin = watermarking_utils.validate_histogram(hist, 3)
    results.append(str(minbin))
    results.append(str(maxbin))
    results.append(str(avgbin))

count = 5
results = np.array(results).reshape(-1, count)

print(results)
print(mean_resamp)
print(mean_ts)

table_data = np.insert(results, len(results[0]), values=mean_resamp, axis=1)
table_data = np.insert(table_data, len(table_data[0]), values=mean_ts, axis=1)
header_row = np.array([["Testfile","$N$ (Samples)", "$min(h(i))$", "$max(h(i))$", "$\\varnothing h(i)$", "$\\varnothing BER$ - resample", "$\\varnothing BER$ - pitch-invariant"]])
table_data = np.append(header_row, table_data, axis=0)

latex_table_utils.print_table(table_data)