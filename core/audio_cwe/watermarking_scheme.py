__author__ = 'gru'

from abc import abstractmethod
from abc import ABCMeta
from os import path
import math

import numpy as np


class WatermarkingSystem(object):
    """An abstract class, which resembles a high-level watermarking system.
    """

    __metaclass__ = ABCMeta

    @classmethod
    def from_file(cls, file):
        if path.exists(file):
            with open(file, 'r') as f:
                kwargs = eval(f.read())
            return cls(**kwargs)
        else:
            raise OSError(2, 'No such file or directory', file)

    @abstractmethod
    def set_params(self, **kwargs):
        """"Set or reset parameters"""

        pass

    @abstractmethod
    def get_params(self):
        """Return the initialization parameters as list"""

        pass

    @abstractmethod
    def embed_watermark(self, data, mark, **kwargs):
        """Return marked media data"""

        pass

    @abstractmethod
    def extract_watermark(self, data, **kwargs):
        """"Return extracted mark"""

        pass

    @staticmethod
    def check_wmk_alignment(num_channels, w):
        """Checks, if the mark fits the channel layout. If there are
        multiple channels, but only one mark (a 1d list
        respectively), it is assumed, that the same mark should be embedded
        in each channel.
        Therefore a 2d representation of the mark is constructed,
        which contains the mark num_channel times.
        (Example: input w = [1, 0, 1], num_channels = 2 -> return wmk = [[1,
        0, 1],[1, 0, 1]])

        :param num_channels: scalar, which specifies the number of channels
        :param w: a list (1d, or 2d), which represents the watermark
        :return: wmk: either the original mark w or a modified version
        """
        wmk = np.array(w)

        if num_channels != len(wmk):
            if wmk.ndim == 1:
                for i in range(0, num_channels - 1):
                    wmk = np.vstack((wmk, w))
                return wmk
            else:
                raise ValueError(
                    'Dimensions of watermark > 1, but does not match '
                    'num_channels')

        else:
            return wmk

    @staticmethod
    def check_key_alignment(num_channels, key):
        """Checks, if the form of the key fits the channel layout. If there
        are multiple channels, but only one key
        (or 1d list), it is assumed, that the same key should be used for in
        each channel. If this is the case, a
        2d represenaton is constructed the contains the key num_channel times.

        :param num_channels: the number of channels
        :param key: scalar or list, which is either a seed or a list of bin
        pairs
        :return: key or modded_key: modified key
        """

        # Precondition the given key to be a Numpy array

        # Scalar - neither list nor np.ndarray
        if not isinstance(key, (list, np.ndarray)):
            key = np.array([key])
        # Scalar stored in np.ndarray
        elif isinstance(key, np.ndarray) and key.size == 1:
            key = np.array([key])
        # Standard python list
        elif isinstance(key, list) and not isinstance(key, np.ndarray):
            key = np.array(key)

        # Handle seeds
        if key.ndim == 1 and num_channels != len(key):
            # Use the same seed for both channels
            if len(key) == 1:
                modded_key = key
                for i in range(num_channels - 1):
                    modded_key = np.hstack((modded_key, key))
                return modded_key
            # Multiple seeds, but not corresponding to the channel layout
            else:
                raise ValueError(
                    'Length of key > 1, but does not match num_channels')

        # Handle bin pairs
        elif key.ndim > 1:

            # Check single bin pair list
            if key.ndim == 2:

                # to fit a mono signal
                if num_channels == 1:
                    return key

                # or duplicate the list to fit a multi channel signal
                else:
                    # Form n-Tuple
                    keys_to_stack = (key for i in range(num_channels))
                    print(keys_to_stack)
                    modded_key = np.stack(keys_to_stack)

                    return modded_key

            # Check that the bin pair list fits the channel layout
            elif key.ndim == 3:
                if len(key) == num_channels:
                    return key

                elif len(key) != 1:
                    # Form n-Tuple
                    keys_to_stack = (key[0] for i in range(num_channels))
                    print(keys_to_stack)
                    modded_key = np.stack(keys_to_stack)
                    return modded_key
                else:
                    raise ValueError(
                        'Length of 3 dimensional key > 1, but does not match '
                        'num_channels')

        return key


class HistBasedWMSystem(WatermarkingSystem):
    """ An abstract histogram based watermarking system
    """
    __metaclass__ = ABCMeta

    @staticmethod
    def mean_of_absolute_values(samples):
        """Calculates the mean of the absolute values of the given samples.

        :param samples: the amplitude values to average
        :return: the mean of absolute values
        """
        if samples.ndim > 1:
            length, num_channels = samples.shape

            mean = np.zeros(num_channels, dtype=np.float)
            for i, frame in enumerate(samples):
                for j, s in enumerate(frame):
                    mean[j] += abs(s)
            mean /= 1.0 * length
        else:
            mean = 0.0
            for i in range(len(samples)):
                mean += abs(samples[i])
            mean /= 1.0 * len(samples)

        return mean

    @staticmethod
    def generate_histogram(samples, la, num_bins, mean):
        """Extracts the histogram of samples with the given amount of bins in
        the amplitude range -la*mean,..,la*mean.

        :param samples: the samples to bin
        :param la: scaling factor
        :param num_bins: number of bins
        :param mean: the given mean of absolute values of the samples. This
        parameter is given explicitly, because in the detection process the
        mean is modified to extract different histograms of the same samples.
        :return: hist, bins: the histogram itself and the bin edges
        """
        assert num_bins <= la * mean * 2, 'num_bins is too large for ' \
                                          'amplitude ' \
                                          'range'
        # generate histogram with integer bins

        # this calculation is necessary to get evenly spaced integer bins
        # [-4544 -4208 ... 4192  4528] otherwise one gets [-4544.
        # -4207.40740741 -3870.81481481...4207.40740741  4544.]
        a_min = np.floor(-la * mean)
        a_max = np.ceil(la * mean)
        step = int(round((abs(a_min) + a_max) / num_bins, 0))

        bins = [int(a_min)]
        for i in range(1, num_bins + 1):
            bins.append(bins[i - 1] + step)

        hist, bins = np.histogram(samples, bins, (a_min, bins[-1]))

        return hist, bins

    @staticmethod
    def generate_standard_histogram(samples, r=16):
        """Generates a histogram with bin width == 1 in the amplitude range
        2**(-r-1)...2**(r-1).

        :param samples: the samples, that are used for forming the histogram
        :param r: the bitdepth of the signal
        :return: hist, bins: the histogram and a list, that specifies the
        bin edges
        """
        bins = np.arange(0, int(2 ** r), 1)
        bins -= 2 ** (r - 1)
        hist, bins = np.histogram(samples, bins)

        return hist, bins

    @staticmethod
    def find_nearest(array, value):
        """Retrieves the index of the element in an array, which has the
        smallest difference to the given value.

        :param array:
        :param value:
        :return: index of the element with the least difference
        """
        return (np.abs(array - value)).argmin()

    @staticmethod
    def find_bin(bins, value):
        """Finds the index of the bin, which contains the given value.

        :param bins: a list, which defines the bin edges
        :param value: the value to 'classify'
        :return: the index of the bin, which contains value
        """
        nearest_idx = HistBasedWMSystem.find_nearest(bins, value)
        if bins[nearest_idx] < value:
            return nearest_idx
        else:
            return nearest_idx - 1