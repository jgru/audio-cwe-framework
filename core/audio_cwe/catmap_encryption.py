__author__ = 'gru'

import numpy as np
from scipy import stats


def apply_catmap(samples, block_size, a, b):
    """Applies Arnold's generalized CatMap to a square 'grid' of samples.
    Resembles
    one
    iteration.

    :param samples: the samples to permute
    :param block_size: effectively the 'scanline', which defines the length
    in the x-dimension
    :param a: one parameter of the CatMap
    :param b: the secon parameter of the CatMap
    :return: cipher_samples: the permuted samples
    """
    cipher_samples = np.zeros(len(samples))
    for y in range(0, block_size):
        for x in range(0, block_size):
            x2 = (x + b * y) % block_size
            y2 = (a * x + (a * b + 1) * y) % block_size
            cipher_samples[y2 * block_size + x2] = samples[y * block_size + x]

    return cipher_samples


def apply_inverse_catmap(cipher_samples, block_size, a, b):
    """Applies the inverse of Arnold's generalized CatMap to a square 'grid' of
    samples. Resembles one iteration.

    :param samples: the samples to permute
    :param block_size: effectively the 'scanline', which defines the length
    in the x-dimension (and y-dim)
    :param a: one parameter of the CatMap
    :param b: the secon parameter of the CatMap
    :return: plain_samples: the inversely permuted samples
    """
    plain_samples = np.zeros_like(cipher_samples)
    for y in range(0, block_size):
        for x in range(0, block_size):
            x2 = ((a * b + 1) * x - b * y) % block_size
            y2 = (y - (a * x)) % block_size
            plain_samples[y2 * block_size + x2] = cipher_samples[
                y * block_size + x]

    return plain_samples


def generate_padding(plain_samples, block_size):
    """Generates padding samples according to the statistics of the signal.
    The padding samples enable the arrangement of the samples on a square
    grid. This is the case, because sqrt(len(play_samples) + block_size)==0.

    :param plain_samples: the samples to be padded
    :param block_size: the blocksize defines the length and height of the grid
    :return:
    """
    padding_length = block_size * block_size - len(plain_samples)

    bincount = (-1 * (np.amin(plain_samples)))
    bincount += (np.amax(plain_samples) + 1)  # add the zero bin
    hist, bins = np.histogram(plain_samples, bincount, density=True)
    xk = np.arange(np.amin(plain_samples), np.amax(plain_samples) + 1)

    pk = hist / np.sum(hist)  # if bin width = 1

    sampdist = stats.rv_discrete(name='Sample Distribution', values=(xk, pk))
    randsamples = sampdist.rvs(size=padding_length)

    return randsamples


def calc_block_size(plain_samples):
    """Calculates the block size, which forms the length and the height of
    the square, on which the padded plain_samples can be arranged. Blocksize is
     chosen, so that sqrt(len(play_samples) + block_size)==0.

    :param plain_samples: the samples, which should be interpreted as a 2d
    square
    :return: block_size: the smallest possible dimension of a square
    """
    block_size = np.sqrt(len(plain_samples))

    if block_size % 1 > 0:
        block_size += 1

    block_size = np.floor(block_size)

    return int(block_size)


def encrypt_mono(plain_samples, a, b, n):
    """Applies Arnold's generalized CatMap wth parameters (a, b) n-times to
    the given samples, which are a single channel.

    :param plain_samples: the plain_samples to be permuted
    :param a: one param of the CatMap
    :param b: the second param of the CatMap
    :param n: the number of iterations
    :return: cipher_samples the permuted samples
    """
    block_size = calc_block_size(plain_samples)
    padding = generate_padding(plain_samples, block_size)
    cipher_samples = np.append(plain_samples,
                               padding)  # axis not given -> flattened
                               # before appending

    for i in range(0, n):
        cipher_samples = apply_catmap(cipher_samples, block_size, a, b)

    cipher_samples = cipher_samples.astype(plain_samples.dtype)

    return cipher_samples, len(padding)


def encrypt(samples, a, b, n):
    """Encrypts a given multi-channel samples by permutation of the
    samples 'positions' with the means of Arnold's generalized CatMap.

    :param samples: the samples to encrypt
    :param a: one param of the CatMap
    :param b: second param of the CatMap
    :param n: number of iterations to perform
    :return: cipher_samples: the encrypted samples
    """
    dt = samples.dtype
    cipher_samples = np.empty(0, dtype=dt)

    if samples.ndim > 1:
        length, num_channels = samples.shape
        for i in range(0, num_channels):
            c_samps, padding = encrypt_mono(samples[:, i], a, b, n)
            cipher_samples = np.append(cipher_samples, c_samps)

        cipher_samples = np.reshape(cipher_samples,
                                    (num_channels, length + padding))
        cipher_samples = cipher_samples.transpose()

    else:
        cipher_samples, padding = encrypt_mono(samples.flatten(), a, b, n)

    return cipher_samples, padding


def decrypt(cipher_samples, padding_length, a, b, n):
    """Decrypts a given multi-channel samples by permutation of the
    samples 'positions' with the means of Arnold's generalized CatMap.

    :param cipher_samples: the encrypted samples, which will be decrypted
    :param padding_length: the length of the padding, which as has to be
    stripped of the end of the plaintext samples.
    :param a: one param of the CatMap
    :param b: second param of the CatMap
    :param n: the number of iterations
    :return: plain_samples: the decrypted samples
    """
    if cipher_samples.ndim > 1:
        length, num_channels = cipher_samples.shape
        plain_samples = np.empty(0, dtype=cipher_samples.dtype)

        for i in range(0, num_channels):
            plain_samples_mono = cipher_samples[:, i]
            block_size = calc_block_size(plain_samples_mono)
            # No further separation as mono, because no padding has to be
            # generated
            for j in range(0, n):
                plain_samples_mono = apply_inverse_catmap(plain_samples_mono,
                                                          block_size, a, b)

            plain_samples = np.append(plain_samples,
                                      plain_samples_mono[:(length - padding_length)])

        plain_samples = np.reshape(plain_samples,
                                   (num_channels, length - padding_length))
        plain_samples = plain_samples.transpose()

    else:
        block_size = calc_block_size(cipher_samples)
        plain_samples = cipher_samples

        for i in range(0, n):
            plain_samples = apply_inverse_catmap(plain_samples, block_size, a,
                                                 b)

        plain_samples = plain_samples[:(len(plain_samples) - padding_length)]

    plain_samples = plain_samples.astype(cipher_samples.dtype)

    return plain_samples


