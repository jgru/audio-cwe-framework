__author__ = 'gru'

import numpy as np
from ast import literal_eval
from os import listdir, makedirs
from os.path import isfile, join, exists

"""
This module house some handy functionality for some repetitive tasks in the
context of watermarking.
"""

def construct_watermark(content):
    """Receives a string or list of strings and forms a binary
    representation of the text as a numpy array.

    :param content: either a string or a list of strings
    :return: a binary representation of the text as a list of uints
    """
    if isinstance(content, list):
        for i, substring in enumerate(content):
            # Construct the mark for channel #i
            bitstring = text_to_bitstring(substring)# form bitstring
            wi = bitstring_to_list(bitstring)  # split it into a list

            if i == 0:
                w = wi
            else:
                w = np.vstack((w, wi))

    else:
        bitstring = text_to_bitstring(content)  # form bitstring
        w = bitstring_to_list(bitstring)  # split it into a list

    return w


def list_to_bitstring(list):
    """Takes a list of zeros and ones and forms a bitstring out of it

    :param list:
    :return: a string representation
    """
    bitstring = 2  # to keep leading zeros start with 0b10 and later strip
    # off the 1

    # Take each list element perform bitwise or and perform left shift
    for i, b in enumerate(list):
        bitstring = bitstring | b
        if i < len(list) - 1:
            bitstring <<= 1

    return bin(bitstring)[3:]  # strip off '0b1'


def bitstring_to_list(bitstring):
    """Takes a string of bits and forms a list out of it.

    :param bitstring: the string of bits
    :return: bits: a list representation of the bitstring
    """
    bits = []

    for i, b in enumerate(bitstring):
        bits.append(int(b))

    return np.array(bits, dtype=np.int)


# http://stackoverflow.com/questions/7396849/convert-binary-to-ascii-and
# -vice-versa
def text_to_bitstring(text, encoding='utf-8', errors='surrogatepass'):
    """Takes text and changes it to its binary representation.

    :param text: a string
    :param encoding: the encoding of the string (utf-8, utf-16, ascii)
    :param errors: specifies how to handle error, when encoding
    :return: the binary representation of the string
    """
    int_rep = int.from_bytes(text.encode(encoding, errors), 'big')
    bits = bin(int_rep)[2:]  # strip off the '0b'
    num_bytes = (len(
        bits) + 7) // 8  # calc the number of whole bytes, // == math.floor()
    bits = bits.zfill(
        8 * num_bytes)  # fill with zeros from left until specified length

    return bits


def bitstring_to_text(bits, encoding='utf-8', errors='surrogatepass'):
    """Takes a bitstring (=binary representation of text) and changes it to
    text according to the specified encoding.

    :param bits: the bitstring
    :param encoding: the encoding of the string
    :param errors: specifies how to handle errors
    :return: the textual representation or nullbyte
    """
    n = int(bits, 2)  # get the integer reperesented by the bitstring
    num_bytes = (
                n.bit_length() + 7) // 8  # calculate the number of bytes,
                # // == math.floor()
    resulting_bytes = n.to_bytes(num_bytes,
                                 'big')  # convert to bytes and get a
                                 # sequence of octets
    text = resulting_bytes.decode(encoding, errors)

    return text or '\0'


def dump_params(out_dir, name_prefix, key=None, iv=None, mark=None):
    """Write parameters of a watermarking process to disk in the specified
    directory.

    :param out_dir: output directory
    :param name_prefix: prefix of the files to write
    :param key: the key (scalar or (multi-dimensional) list
    :param iv: the parameters for the WM system (initialization vector)
    :param mark: the mark
    :return: None
    """
    if key is not None:
        # Ensure, that key is a Numpy ndarray
        if isinstance(key, (list, np.ndarray)):
            key_arr = np.array(key)
        else:
            key_arr = np.array([key])

        if key_arr.ndim > 2:
            np.savetxt(join(out_dir, name_prefix + '_key_' + str(key.shape)),
                       key.flatten(), fmt='%i')
        else:
            np.savetxt(join(out_dir, name_prefix + '_key'), key_arr, fmt='%i')

    if iv is not None:
        with open(join(out_dir, name_prefix + '_iv'), 'w') as f:
            f.write(str(iv))

    if mark is not None:
        np.savetxt(join(out_dir, name_prefix + '_mark'), np.array(mark),
                   fmt='%i')


def read_keyfile(filepath):
    """Read a key file, that stores bin pairs. If it ends with a tuple,
    then specifies than the key is stored in a one dimensional manner and
    the tuple specifies the shape of the key.

    :param filepath: the path to the key file to read
    :return key: the extracted key
    """
    key = np.loadtxt(filepath, dtype=np.uint32)

    suffix = filepath.rsplit('_', maxsplit=1)[-1]
    if suffix.startswith('(') and suffix.endswith(')'):
        # Restore it correctly
        shape = literal_eval(
            suffix)  # cut off the appended tuple and interpret it as one
        key = key.reshape(shape)

    return key


def compare_watermarks(w1, w2, is_print=True):
    print('Original mark:\n', w1)
    print('Recovered mark:\n',w2)

    ber = calc_bit_error_rate(w1, w2)
    if np.all(ber == 0): #ber == 0.0
        print('Original and recovered mark match perfectly.')
        print('BER: ', ber)
    else:
        print('Original and recovered mark do not match.')
        print('BER: ', ber)
        return ber


def calc_bit_error_rate(b1, b2):
    """Calculates the bit error rate for two binary lists.

    :param b1: a list of bits
    :param b2: another list of bits
    :return: the resulting bit error rate
    """
    if len(b1) != len(b2):
        raise ValueError('List of bits differ in length')

    # calc difference element wise
    diff = np.array(b1) - np.array(b2)

    # sum up the absolute values to get the amount of bit errors
    if diff.ndim == 1:
        result = np.sum(np.absolute(diff))

        # divide the amount of bit errors by total amount of transferred bits
        return result / len(diff)
    else:
        result = np.sum(np.absolute(diff), axis=1)

        # divide the amount of bit errors per channel by amount of
        # transferred bits per channel
        return result / np.size(diff, 1)



def snr(samples_a, samples_b):
    """Calculates the signal-to-noise-ratio in dB.

    :param samples_a: signal a
    :param samples_b: signal b
    :return: the SNR in dB
    """
    # Convert to 64-bit integers to prevent potential integer overflow
    samples_a = samples_a.astype(np.int64)
    samples_b = samples_b.astype(np.int64)

    # Calculates MSE
    noise = np.sum((samples_a - samples_b) ** 2)

    if noise == 0:
        return np.inf

    return 10 * np.log10(np.sum(samples_a ** 2) / noise)


def psnr(samples_a, samples_b):
    """Calculates PSNR in dB.

    :param samples_a: singal a
    :param samples_b: signal b
    :return: the PSNR in db
    """
    # Gather maximum amplitude in signal a
    if samples_a.dtype == np.float32:
        peak = 1.0
    elif samples_a.dtype == np.int16:
        peak = 2 ** 15

    # Convert to 64-bit integers to prevent potential integer overflow
    samples_a = samples_a.astype(np.int64)
    samples_b = samples_b.astype(np.int64)

    # Calculates MSE
    noise = np.mean((samples_a - samples_b) ** 2)

    if noise == 0:
        return np.inf

    return 10 * np.log10(peak ** 2 / noise)


def validate_histogram(hist, factor):
    """Prints some statistics about the histogram and checks, that h(k)>>L.
    (hist(k)>factor*len(hist))

    :param hist: the histogram
    :param factor: a factor, that scales the length of the histogramm
    :return: the bin with the minimal content, the maximal one and the average
    """
    count = 0
    sum = 0
    for i, hi in enumerate(hist):
        sum += hi
        if hi < factor * len(hist):
            count += 1

    print('Average amount of samples in bins: ', round(sum / len(hist)))
    print('Min bin: ', np.amin(hist))
    print('Max bin: ', np.amax(hist))
    print('Count of bins h(i)<<L: ', count)

    return np.amin(hist), np.amax(hist), round(sum / len(hist))