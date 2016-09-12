__author__ = 'gru'

from os import listdir, path

import soundfile as sf
import numpy as np

from core.audio_cwe import watermarking_utils
from core.audio_cwe.xs_wm_scheme import XsWMSystem
from core.audio_cwe import catmap_encryption
from experimental_testing import latex_table_utils


'''
input_dir = '../../res/testing/original_test_files/selection'
samples, samplerate = sf.read(input_dir + '/' + '46.wav', dtype=np.int16)
samples= samples[:, 0]
# Construct watermarking system
wm_sys = XsWMSystem(la=la, num_bins=num_bins, threshold=bin_threshold,
                    step=step)
wmk = watermarking_utils.construct_watermark('Q')
# Embed mark
marked_samples, bp = wm_sys.embed_watermark(samples, wmk, key=68432)
params = wm_sys.get_params()
print(bp)
# Verify the mark
w_2 = wm_sys.extract_watermark(marked_samples, key=bp, delta=0.0, syn=wmk[:4])
print(wmk)
print(w_2)
if np.array_equal(wmk, w_2):
    print('True')
'''


def check_emv(input_dir, wav_files):
    results = []

    for i, f in enumerate(wav_files):
        print(f)
        samples, samplerate = sf.read(input_dir + '/' + f, dtype=np.int16)
        samples = samples[:, 0]

        # Encrypt the signal
        ciphered_samples, padding_length = catmap_encryption.encrypt(samples,
            a, b, n)

        # Choose seed pseudo-randomly
        seed = int(
            (max_seed - min_seed) * np.random.random_sample() + min_seed)

        # Generate random mark
        wmk = []
        for c in range(wmk_len):
            wmk.append(round(np.random.sample()))

        # Construct watermarking system
        wm_sys = XsWMSystem(la=la, num_bins=num_bins, threshold=bin_threshold,
                            step=step)

        # Embed mark
        marked_samples, bp = wm_sys.embed_watermark(ciphered_samples, wmk,
                                                    key=seed)

        # Verify the mark
        w_2 = wm_sys.extract_watermark(marked_samples, key=bp, delta=0.1,
                                       syn=wmk[:4])
        print(wmk)
        print(w_2)
        print('BER: ', watermarking_utils.calc_bit_error_rate(wmk, w_2))
        results.append(watermarking_utils.calc_bit_error_rate(wmk, w_2))
    return results


def check_mev(input_dir, wav_files):
    results = []

    for i, f in enumerate(wav_files):
        print(f)
        samples, samplerate = sf.read(input_dir + '/' + f, dtype=np.int16)
        samples = samples[:, 0]

        # Choose seed pseudo-randomly
        seed = int(
            (max_seed - min_seed) * np.random.random_sample() + min_seed)

        # Generate random mark
        wmk = []
        for c in range(wmk_len):
            wmk.append(round(np.random.sample()))

        # Construct watermarking system
        wm_sys = XsWMSystem(la=la, num_bins=num_bins, threshold=bin_threshold,
                            step=step)

        # Embed mark
        marked_samples, bp = wm_sys.embed_watermark(samples, wmk, key=seed)

        # Encrypt the signal
        ciphered_samples, padding_length = catmap_encryption.encrypt(
            marked_samples, a, b, n)

        # Verify the mark
        w_2 = wm_sys.extract_watermark(ciphered_samples, key=bp, delta=0.0,
                                       syn=wmk[:4])
        print(wmk)
        print(w_2)
        print('BER: ', watermarking_utils.calc_bit_error_rate(wmk, w_2))
        results.append(watermarking_utils.calc_bit_error_rate(wmk, w_2))

    return results


def check_medv(input_dir, wav_files):
    results = []

    for i, f in enumerate(wav_files):
        print(f)
        samples, samplerate = sf.read(input_dir + '/' + f, dtype=np.int16)
        samples = samples[:, 0]

        # Choose seed pseudo-randomly
        seed = int(
            (max_seed - min_seed) * np.random.random_sample() + min_seed)

        # Generate random mark
        wmk = []
        for c in range(wmk_len):
            wmk.append(round(np.random.sample()))

        # Construct watermarking system
        wm_sys = XsWMSystem(la=la, num_bins=num_bins, threshold=bin_threshold,
                            step=step)

        # Embed mark
        marked_samples, bp = wm_sys.embed_watermark(samples, wmk, key=seed)

        # Encrypt the signal
        ciphered_samples, padding_length = catmap_encryption.encrypt(
            marked_samples, a, b, n)

        # Decrypt signal
        decrypted_samples = catmap_encryption.decrypt(ciphered_samples,
                                                      padding_length, a, b, n)

        # Verify the mark
        w_2 = wm_sys.extract_watermark(decrypted_samples, key=bp, delta=0.0,
                                       syn=wmk[:4])
        print(wmk)
        print(w_2)
        print('BER: ', watermarking_utils.calc_bit_error_rate(wmk, w_2))
        results.append(watermarking_utils.calc_bit_error_rate(wmk, w_2))

    return results


input_dir = '../../res/testing/original_test_files/selection'
# retrieve files
files = [f for f in listdir(input_dir) if path.isfile(path.join(input_dir, f))]
wav_files = [f for f in files if f.endswith('.wav')]

# Read the file labels
labels = ['']  # empty string at pos one is necessary for displaying table
with open(path.join(input_dir, 'labels'), 'r') as f:
    for line in f.readlines():
        labels.append(line.replace('\n',''))

# some watermarking parameters
max_seed = 100000
min_seed = 30

wmk_len = 32
bin_multiplier = 4
num_bins = 128
la = 2.5
bin_threshold = 25
step = 5

a = 20
b = 32
n = 2

results_mev = check_mev(input_dir, wav_files)
results_emv = check_emv(input_dir, wav_files)
results_medv = check_medv(input_dir, wav_files)

print('------------------------------------------')
print('V(E(M(O,m)))')
second_row = np.append(['BER'], results_mev)
table_data = np.vstack((labels, second_row))
latex_table_utils.print_table(table_data)

print('------------------------------------------')
print('V(M(E(O),m))')
second_row = np.append(['BER'], results_emv)
table_data = np.vstack((labels, second_row))
latex_table_utils.print_table(table_data)

print('------------------------------------------')
print('V(D(M(E(O),m))')
second_row = np.append(['BER'], results_medv)
table_data = np.vstack((labels, second_row))
latex_table_utils.print_table(table_data)




