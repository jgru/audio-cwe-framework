__author__ = 'gru'

import numpy as np
import scipy.spatial.distance

from core.audio_cwe.xiang_wm_scheme import XiangWMSystem
from core.audio_cwe.schmitz_wm_scheme import SchmitzWMSystem


class XsWMSystem(SchmitzWMSystem, XiangWMSystem):
    """
    This class implements a combined and modified version of Xiang et al.'s and
    Schmitz et al.'s watermarking methods. At first the amplitude histogram in
    a certain amplitude range is formed. A watermark is then embedded by
    considering the relation of pairs of bins (a_i, b_i), which are
    pseudo-randomly selected. The embedding key is the
    seed and the detection key is a list of bin pairs. Embedding the i-th
    bit is realized
    by swapping the i-th bin pair, if the relation resembles not the wanted
    proportion.
    """

    # Specifies the additional keys for the parameter dictionary,
    # that supplement the param keys from the superclasses
    PARAM_KEYS = ['orig_mean']

    def __init__(self, la=2.0, num_bins=128, threshold=50, delta=0.05,
                 step=(-9, 9), orig_mean=None):
        """Constructs and initializes a XsWMSystem object.

        Keyword arguments:
        :param la: a scalar, which specifies lambda -  a factor which
        influences the amplitude range for hist extraction
        :param num_bins: a scalar, which specifies the number of histogram
        bins to extract from the amplitude range
        :param threshold: a scalar, which specifies the absolute difference,
        that bins must exhibit to can be considered
                for forming a bin pair
        :param delta: a scalar >0.0, which specifies the size of the search
        space formed by extraction
        :param step: a tuple or scalar, which specifies the step size
        :param orig_mean: scalar or list, which specifies the original mean
        of the signal before embedding a mark
                for each channel
        :return: None
        """
        self._orig_mean = None

        # Call superclasses for initialization
        XiangWMSystem.__init__(self, la=la, num_bins=num_bins,
                               threshold=threshold, delta=delta)
        SchmitzWMSystem.__init__(self, step=step)

        self.set_params(orig_mean=orig_mean)

    def set_params(self, **kwargs):
        """Sets the parameters of the watermarking system. This is a
        convenience method, that  uses **kwargs,
        so that it's possible to set a multiple parameters without changing
        the others to default

        Keyword arguments:
        :param la: a scalar, which specifies lambda -  a factor which
        influences the amplitude range for hist extraction
        :param num_bins: a scalar, which specifies the number of histogram
        bins to extract from the amplitude range
        :param threshold: a scalar, which specifies the absolute difference,
        that bins must exhibit to can be considered
                for forming a bin pair
        :param delta: a scalar >0.0, which specifies the size of the search
        space formed by extraction
        :param step: a tuple or scalar, which specifies the step size
        :param orig_mean: scalar or list, which specifies the original mean
        of the signal before embedding a mark
                for each channel
        :return: None
        """

        # Delegate the parameter setting to superclasses
        XiangWMSystem.set_params(self, **kwargs)
        SchmitzWMSystem.set_params(self, **kwargs)

        # Only necessary for detection
        if 'orig_mean' in kwargs:
            if kwargs['orig_mean'] is not None:
                self.orig_mean = kwargs['orig_mean']

        self._is_init = True

    def get_params(self):
        """Returns all parameters as a dictionary

        :return: params: a dict containing all parameters
        """
        # Delegate the retrieval of the parameters to superclasses
        params = XiangWMSystem.get_params(self).copy()
        params.update(SchmitzWMSystem.get_params(self))

        # Add orig_mean if set, only necessary for detection
        if self.orig_mean is not None:
            d = dict(zip(XsWMSystem.PARAM_KEYS, [self.orig_mean]))
            params.update(d)

        return params

    @property
    def orig_mean(self):
        return self._orig_mean

    @orig_mean.setter
    def orig_mean(self, value):
        if value is not None:
            if isinstance(value, (tuple, list)):
                for i, v in enumerate(value):
                    if v <= 0:
                        raise ValueError('Mean value not valid')
            else:
                if value <= 0:
                    raise ValueError('Mean value not valid')

            self._orig_mean = value

    def embed_watermark(self, samples, w, **kwargs):
        """Embeds the specified mark in the samples.

        :param samples: the signal to be marked
        :param w: the watermark
        :param kwargs: the embedding key (in this case the seed)
        :return: None
        """

        assert self._is_init, 'WM system NOT initialized'
        self.set_params(**kwargs)

        if self._num_bins < 2 * len(w):
            raise ValueError('Not enough bins for length of mark')

        if 'key' in kwargs:
            seed = kwargs['key']
        else:
            raise TypeError('Required parameter \'key\'(seed) is missing')

        print('=============================================')
        print("Embedding ", w, " via combined method")
        print('---------------------------------------------')

        # make a deep copy of the samples to mark
        samples_to_mark = np.empty_like(samples)
        samples_to_mark[:] = samples
        # Check, if multi channel audio has to be marked
        if samples.ndim > 1:
            length, num_channels = samples.shape
            wmk = self.check_wmk_alignment(num_channels, w)
            seed = self.check_key_alignment(num_channels, seed)

            # Pre-calculate the original mean values
            self._orig_mean = []
            for i in range(0, num_channels):
                self._orig_mean.append(
                    self.mean_of_absolute_values(samples[:, i]))

            # Embed the mark(s)
            bin_pairs = np.array([])
            for i in range(0, num_channels):
                print(self.orig_mean[i])
                print('in channel #', i)
                print('---------------------------------------------')
                samples_to_mark[:,
                i], bp = self._embed_watermark_single_channel(
                    samples_to_mark[:, i], wmk[i], seed[i], self.orig_mean[i])
                if i == 0:
                    bin_pairs = bp
                else:
                    bin_pairs = np.stack((bin_pairs, bp), axis=0)

            return samples_to_mark, bin_pairs

        else:
            print('in channel #0')
            print('---------------------------------------------')
            self._orig_mean = self.mean_of_absolute_values(samples)

            return self._embed_watermark_single_channel(samples, w, seed,
                                                        self._orig_mean)

    def _embed_watermark_single_channel(self, samples_to_mark, w, seed, mean):
        """Embeds the watermark in a mono signal.

        :param samples_to_mark: the samples to mark
        :param w: the watermark
        :param seed: the key - more precise: the seed for the PRNG
        :param mean: the original mean of the signal
        :return: marked_samples, bin_pairs: marked copy of the samples and
        the bin pairs (detection key)
        """
        assert self._num_bins > 2 * len(
            w), "Number of bins < 2*the length of the bitstring to embed"

        hist, bins = self.generate_histogram(samples_to_mark, self.la,
                                             self._num_bins, mean)

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

        This is for histograms with less bins more efficient, that the
        superclass method, which is overridden.

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
            b = XsWMSystem.find_bin(bins, x)

            # Get (row_ids, col_ids)-tuple via where()
            item_index = np.where(bins_to_swap == b)

            if len(item_index[0]) > 0:  # if at least one row index was found
                ids = bins_to_swap[item_index[0][0]]
                if item_index[1][0] == 0:
                    samples_to_mark[i] += ((ids[1] - ids[0]) * bin_width)
                if item_index[1][0] == 1:
                    samples_to_mark[i] -= ((ids[1] - ids[0]) * bin_width)

        return samples_to_mark

    MARGIN = 2

    def _gen_pair(self, hist, bins, used_bins):
        """Overrides the method of the superclass SchmitzWMSystem to add
        threshold checking.

        :param hist: histogram of signal
        :param bins: an array, that specifies the bin edges
        :param used_bins: an array, that contains already used bins
        :return: a tuple of the pair's indices
        """

        idx_a = self._gen_a_i(hist, bins, used_bins)

        step = self._prng.randint(max(self.min_step, 0 - idx_a),
                                  min((len(hist) - XsWMSystem.MARGIN) - idx_a,
                                      self.max_step) + 1)
        idx_b = idx_a + step

        if (idx_b not in used_bins) and (
                    abs(hist[idx_a] - hist[idx_b]) > self.threshold):
            used_bins.append(idx_b)
            return idx_a, idx_b
        else:
            used_bins.remove(idx_a)
            # print('Regenerate pair: ', a_i, ', ', b_i)

            return self._gen_pair(hist, bins, used_bins)

    def extract_watermark(self, samples, **kwargs):
        """Extracts the watermark from the given samples by means of the
        given key.

        :param samples: the marked samples
        :param kwargs: various parameters; key, syn are required, orig_mean
        had to be set
        :return: w: the extracted mark (if samples is a multi-channel
        signal, then a list of marks is returned)
        """

        # possibly change the parametrization, this is optional
        self.set_params(**kwargs)

        if 'key' in kwargs:
            key = kwargs['key']
        else:
            raise TypeError('Required parameter \'key\' is missing')

        if 'syn' in kwargs:
            syn = kwargs['syn']
        else:
            raise TypeError('Required parameter \'syn\' is missing')

        if 'orig_mean' in kwargs:
            self.orig_mean = kwargs['orig_mean']

        elif self._orig_mean is None:
            raise TypeError('Required parameter \'orig_mean\' is missing')

        print('=============================================')
        print("Detecting watermark")
        print('---------------------------------------------')

        # Handle multi channel signals
        if samples.ndim > 1:
            length, num_channels = samples.shape
            syn = self.check_wmk_alignment(num_channels, syn)
            key = self.check_key_alignment(num_channels, key)

            w = np.empty((num_channels, 1), dtype=np.int)

            for i in range(0, num_channels):
                print('in channel #', i)
                print('---------------------------------------------')
                w_i = self._extract_watermark_single_channel(samples[:, i],
                                                             key[i], syn[i],
                                                             self.orig_mean[i])

                # store extracted watermark in multi-dimensional output array
                if i == 0:
                    w = w_i
                else:
                    w = np.vstack((w, w_i))

        # mono signal
        else:
            print('in channel #0')
            print('---------------------------------------------')
            w = self._extract_watermark_single_channel(samples, key, syn,
                                                       self.orig_mean)

        return w

    def _extract_watermark_single_channel(self, samples, bin_pairs, syn,
                                          orig_mean):
        """Extracts watermark from a marked mono signal.

        :param samples: the marked single channel signal
        :param bin_pairs: the extraction key - specifying the bin pairs to
        compare. Each of the forms a single bit
        :param syn: the synchronization code (part of the watermark
        :param orig_mean: the mean before embedding the mark, this is the
        synchronization point
        :return: w: the extracted mark
        """
        # The mark to extract
        w = []

        # Step through the search space and calculate best match
        if self.delta > 0.0:
            search_space = np.arange(orig_mean * (1 - self.delta),
                                     orig_mean * (1 + self.delta),
                                     1 / self.la)

            best_match = 0
            distance = 1.0
            for j, m in enumerate(search_space):
                extracted = []
                hist, bins = self.generate_histogram(samples, self.la,
                                                     self.num_bins,
                                                     search_space[j])

                for i, p in enumerate(bin_pairs):

                    id1 = p[0]
                    id2 = p[1]

                    if hist[id1] <= hist[id2]:
                        extracted.append(int(1))

                    elif hist[id1] > hist[id2]:
                        extracted.append(int(0))

                extracted_syn = extracted[0:len(syn)]

                # Calculates (2*(n_01 + n10))/(n_00+n_11+2*(n_01 + n10))
                current_distance = scipy.spatial.distance.rogerstanimoto(
                    extracted_syn, syn)
                if current_distance < distance:
                    distance = current_distance
                    best_match = j
                    w = extracted
                elif current_distance == distance:
                    if abs(search_space[j] - orig_mean) <= abs(
                                    search_space[best_match] - orig_mean):
                        distance = current_distance
                        best_match = j
                        w = extracted

            print('---------------------------------------------')
            print('Syn:\n', syn)
            print('Best match:\n', np.array(w[0:len(syn)]))
            print('Mean: ', search_space[best_match])
            print('---------------------------------------------')

        # delta = 0.0, if no searching necessary (attacking can be excluded)
        else:
            hist, bins = self.generate_histogram(samples, self.la,
                                                 self.num_bins, orig_mean)

            for i, p in enumerate(bin_pairs):
                id1 = p[0]
                id2 = p[1]

                if hist[id1] < hist[id2]:
                    w.append(1)

                elif hist[id1] > hist[id2]:
                    w.append(0)

        return np.array(w)