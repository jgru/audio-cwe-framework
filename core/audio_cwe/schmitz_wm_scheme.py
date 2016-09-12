__author__ = 'gru'

import numpy as np
from numpy.random import RandomState

from core.audio_cwe.watermarking_scheme import HistBasedWMSystem


class SchmitzWMSystem(HistBasedWMSystem):
    """
    This class implements Schmitz et al.'s watermarking method, which is
    adapted for the use with audio data.

    A watermark is embedded by forming the histogram of the samples'
    amplitudes and considering the relation of pairs
    of bins (a_i, b_i), which are pseudo-randomly selected. The embedding
    and detection key is the seed for the PRNG.
    The embedding of the i-th bit is realized by swapping the i-th bin pair,
    if the relation resembles not the
    wanted proportion.
    """

    # Class constants specifing minimal and maximal step size
    MIN_STEP = -9
    MAX_STEP = 9

    # Specifies the keys for the parameter dictionary
    PARAM_KEYS = ['step']

    def __init__(self, step=(-9, 9)):
        """Constructs and initializes a SchmitzWMSystem object.

        Keyword arguments:
        :param step: a tuple or scalar, which specifies the step size
        :return: None
        """
        self._min_step = None
        self._max_step = None
        self._prng = None
        self.set_params(step=step)

        self._is_init = True

    @property
    def min_step(self):
        return self._min_step

    @min_step.setter
    def min_step(self, value):
        if value < SchmitzWMSystem.MIN_STEP or value >= 0:
            raise ValueError('Step exceeds the suggested range')
        self._min_step = value

    @property
    def max_step(self):
        return self._max_step

    @max_step.setter
    def max_step(self, value):
        if value <= 0 or value > SchmitzWMSystem.MAX_STEP:
            raise ValueError('Step exceeds the suggested range')
        self._max_step = value

    def set_params(self, **kwargs):
        """Sets the parameters of the watermarking system. This is the
        implementation of an abstract convenience method,
        which is specified by a superclass and uses **kwargs, so that it's
        possible to set a multiple parameters
        without changing the others to default (in this case kind of redundant.

        Keyword arguments:
        :param step: a tuple or scalar, which specifies the step size
        :return: None
        """
        if 'step' in kwargs:
            step = kwargs['step']
            if not isinstance(step, tuple):
                self.min_step = abs(step) * -1
                self.max_step = abs(step)
            else:
                self.min_step, self.max_step = step

        # Init PRNG
        self._prng = RandomState()

    def get_params(self):
        """Returns all parameters as a dictionary

        :return: params: a dict containing all parameters
        """
        return dict(
            zip(SchmitzWMSystem.PARAM_KEYS, [(self.min_step, self.max_step)]))

    def embed_watermark(self, samples, w, **kwargs):
        """Embeds the specified mark in the samples.

        :param samples: the signal to be marked
        :param w: the watermark
        :param kwargs: the embedding key (in this case the seed)
        :return: None
        """

        assert self._is_init, 'WM system NOT initialized'

        if 'key' in kwargs:
            # Retrieve seed from kwargs
            seed = kwargs['key']
        else:
            raise TypeError('Required parameter \'key\'(seed) is missing')

        print('=============================================')
        print('Embedding ', w, ' via Schmitz\' method')
        print('---------------------------------------------')

        # Make a deep copy of the samples to mark
        samples_to_mark = np.empty_like(samples)
        samples_to_mark[:] = samples

        # Check, if multi channel audio has to be marked
        if samples.ndim > 1:
            length, num_channels = samples.shape
            # Check if watermark matches channel layout
            wmk = self.check_wmk_alignment(num_channels, w)
            seed = self.check_key_alignment(num_channels, seed)

            bin_pairs = np.array([])
            for i in range(0, num_channels):
                print('in channel #', i)
                print('---------------------------------------------')

                samples_to_mark[:, i], bp = self._embed_watermark_single_channel(
                    samples_to_mark[:, i], wmk[i], seed[i])  # returns copy
                if i == 0:
                    bin_pairs = bp
                else:
                    bin_pairs = np.stack((bin_pairs, bp), axis=0)

            return samples_to_mark, bin_pairs

        else:
            print('in channel #0')
            print('---------------------------------------------')
            return self._embed_watermark_single_channel(samples_to_mark, w,
                                                       seed)

    def _embed_watermark_single_channel(self, samples_to_mark, w, seed):
        """Embeds the watermark in a mono signal.

        :param samples_to_mark: the samples to mark
        :param w: the watermark
        :param seed: the key - more precise: the seed for the PRNG
        :param mean: the original mean of the signal
        :return: marked_samples, bin_pairs: marked copy of the samples and
        the bin pairs (detection key)
        """
        hist, bins = self.generate_standard_histogram(samples_to_mark)

        # Construct a sequence of pseudo-randomly selected bin pairs
        self._prng.seed(seed)
        bin_pairs = self._generate_bin_pairs(hist, bins, len(w), seed)

        bins_to_swap = []
        for i, bit in enumerate(w):
            id1 = bin_pairs[i][0]
            id2 = bin_pairs[i][1]

            if bit == 1:
                if hist[id1] < hist[id2]:
                    # do nothing
                    continue
                else:
                    # Store the bins to swap
                    bins_to_swap.append((id1, id2))

            if bit == 0:
                if hist[id1] > hist[id2]:
                    continue
                else:
                    # Store the bins to swap
                    bins_to_swap.append((id1, id2))

        # Swap the bins
        marked_samples = self.swap_bins_at_once(samples_to_mark, bins,
                                                bins_to_swap)

        return marked_samples, bin_pairs

    @staticmethod
    def swap_bins_at_once(samples, bins, bins_to_swap):
        """Swaps the bin pairs, that are specified by bins_to_swap,
        which contains the bin indices.
        If a pair (id1, id2) has to be swapped, every sample x, which falls
        in bin[id1] is modified, so that it falls
        into bin[id2].

        :param samples: the samples to be modified
        :param bins: the bin edges
        :param bins_to_swap: a list of pairs to be swapped
        :return:samples_to_mark: the modified samples
        """
        bin_width = abs(bins[1] - bins[0])

        # make a deep copy of the samples to mark
        samples_to_mark = np.empty_like(samples)
        samples_to_mark[:] = samples

        for i, x in enumerate(samples):
            for j, ids in enumerate(bins_to_swap):
                if bins[ids[0]] <= x < bins[ids[0] + 1]:
                    samples_to_mark[i] += ((ids[1] - ids[0]) * bin_width)
                    break
                elif bins[ids[1]] <= x < bins[ids[1] + 1]:
                    samples_to_mark[i] -= ((ids[1] - ids[0]) * bin_width)
                    break

        return samples_to_mark

    def extract_watermark(self, samples, **kwargs):
        """Extracts the watermark from the given samples by means of the
        given key.

        :param samples: the marked samples
        :param kwargs: various parameters; key, syn are required, orig_mean
        had to be set
        :return: w: the extracted mark (if samples is a multi-channel
        signal, then a list of marks is returned)
        """
        assert self._is_init, 'WM system NOT initialized'
        if 'key' in kwargs:
            key = kwargs['key']
        else:
            raise TypeError('Required parameter \'key\' is missing')

        if 'length' in kwargs:
            wmk_len = kwargs['length']
        else:
            raise TypeError('Required parameter \'length\' is missing')

        print('=============================================')
        print("Detecting watermark")
        print('---------------------------------------------')

        # Multi-channel signal
        if samples.ndim > 1:
            length, num_channels = samples.shape
            # Check, that for each channel exist a seed (or use the same for
            #  all)
            key = self.check_key_alignment(num_channels, key)
            w = np.empty((num_channels, 1), dtype=np.int)

            for i in range(0, num_channels):
                print('in channel #', i)
                print('---------------------------------------------')

                # Extract watermark
                w_i = self.extract_watermark_single_channel(samples[:, i],
                                                            key[i], wmk_len)

                # Store extracted watermark in 2d output array
                if i == 0:
                    w = np.array(w_i)
                else:
                    w = np.vstack((w, w_i))

        # Mono signal
        else:
            w = self.extract_watermark_single_channel(samples, key, wmk_len)

        return w

    def extract_watermark_single_channel(self, samples, key, length):
        """Extracts watermark from a marked mono signal.

        :param samples: the marked single channel signal
        :param key: the extraction key
        :param length: the number of bits to extract
        :return: w: the extracted mark
        """
        hist, bins = self.generate_standard_histogram(samples)

        # Construct the same sequence of pseudo-randomly selected bin pairs
        # as on the embedder side
        self._prng.seed(key)
        bin_pairs = self._generate_bin_pairs(hist, bins, length, key)

        w = []
        for i, p in enumerate(bin_pairs):
            id1 = p[0]
            id2 = p[1]

            if hist[id1] < hist[id2]:
                w.append(1)

            elif hist[id1] > hist[id2]:
                w.append(0)

        return np.array(w)

    def _gen_a_i(self, hist, bins, used_bins):
        """Generates the initial bin a for the i_th bit to embed.

        :param hist: The histogram of the signal (necessary to check for bin
        equality or emptiness
        :param bins: A list which specifies the edges of the histogram bins
        :param used_bins: A list of already used bins, that cannot be
        considered anymore
        :return:
        """
        a_i = self._prng.randint(0, len(
            bins) - 1)

        while a_i in used_bins or hist[a_i] == 0:
            if a_i < len(bins) - 2:
                a_i += 1
            else:
                return self._gen_a_i(hist, bins, used_bins)

        used_bins.append(a_i)
        return a_i

    def _gen_pair(self, hist, bins, used_bins):
        """ Generates a single bin pair.

        :param hist: The histogram of the signal (necessary to check for bin
        equality or emptiness
        :param bins: A list which specifies the edges of the histogram bins
        :param used_bins: a list of already used bins, that cannot be
        considered anymore
        :return:
        """
        a_i = self._gen_a_i(hist, bins, used_bins)
        step = self._prng.randint(max(self.min_step, 0 - a_i),
                                  min(len(bins) - 1 - a_i, self.max_step) + 1)
        b_i = a_i + step

        if b_i not in used_bins and (hist[a_i] != hist[b_i]) and hist[b_i] > 0:
            used_bins.append(b_i)
            return a_i, b_i
        else:
            used_bins.remove(a_i)
            return self._gen_pair(hist, bins, used_bins)

    def _generate_bin_pairs(self, hist, bins, length, seed):
        """Constructs the bin pairs, whose relation is used to encode one
        watermark bit. This is done by using a seeded
        PRNG.

        :param hist: The histogram of the signal (necessary to check for bin
        equality or emptiness
        :param bins: A list which specifies the edges of the histogram bins
        :param length: The length of the watermark to be embedded
        :param seed: The seed for the PRNG
        :return: bin_pairs: A 2d list, which contains the drawn bin pairs
        """
        self._prng = RandomState()
        self._prng.seed(seed)

        bin_pairs = np.empty((0, 2), dtype=np.int)
        ub = []

        for i in range(0, length):
            a_i, b_i = self._gen_pair(hist, bins, ub)
            bin_pairs = np.append(bin_pairs, [[a_i, b_i]], axis=0)

        return bin_pairs