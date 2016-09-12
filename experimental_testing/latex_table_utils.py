__author__ = 'gru'

import numpy as np


# based on http://myvirtualbrain.blogspot.de/2014/01/python-array-to-latex-tabular.html
def list2latex(l):
    # Check if list contains enough values
    assert len(l) > 0
    assert len(l[0]) > 0

    # Determine the maximum string length in each column
    lengths = [[len(str(x))for x in row] for row in l]

    max_len = np.array(lengths).max(axis=0)
    result = [[None for col in range(len(l[0]))] for row in range(len(l))]

    # justify the strings
    for x, row in enumerate(l):
        ## justify the columns to align nicely
        for y, item in enumerate(row):
            result[x][y] = item.rjust(max_len[y])

        # Put into latex tabular row format
        result[x] = " & ".join(result[x])
        # Add double backslash at the end
        # Need to escape it in string
        result[x] += "  \\\\"
    return result

def print_table(table_data):
    for row in list2latex(table_data):
        print(row)