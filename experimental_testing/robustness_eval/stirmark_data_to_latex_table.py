__author__ = 'gru'

import numpy as np
from os.path import isfile, isdir, join, split, exists
from experimental_testing import latex_table_utils


input_dir = '../../../res/testing/robustness_eval/32_bits_step_5_t_50/stirmark'
attacks = np.genfromtxt(join(input_dir, 'attack_order'), dtype='str')  # alternative attacks = [:,0]
num_test_files = len(np.unique([s.split('_')[0] for s in attacks]))
#attacks = np.unique([(s.split('-', maxsplit=2)[-1]).split('.')[0] for s in attacks])

ber_values = np.genfromtxt(join(input_dir, 'v2_BER_stirmark_attacks_32_bits'),
                           dtype='str')[:,1].astype(np.float)
ber_values = [round(x,3) for x in ber_values]
order = np.arange(0, len(attacks), 1)
print(len(ber_values)/len(order))
print(len(order))
print(len(attacks))
header_row = np.array([["Index", "Angriff+Parameter", "Tenor singing", "Female speech", "Guitar", "Soloists (Verdi)", "Choir (Orff)", "Abba", "Eddie Rabbit"]])
combined = np.hstack((attacks, ber_values))
combined = np.hstack((order, combined))
print(combined)
table_data = combined.reshape(-1, len(attacks)).transpose()
table_data = np.append(header_row, table_data, axis=0)

latex_table_utils.print_table(table_data)
