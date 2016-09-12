__author__ = 'gru'

import numpy as np
import scipy.spatial.distance

from core.audio_cwe.watermarking_scheme import HistBasedWMSystem


class XiangWMSystem(HistBasedWMSystem):
    """
    This class implements Xiang et al.'s watermarking method for the use
    with audio data.
    It embeds a mark by extraction the histogram from a certain amplitude
    range (formed by lambda and the mean of
    absolute values) and embeds the i-th watermark bit by considering the
    relation of the three consecutive bins
    at index i*3.
    """

    # Specifies the keys for the parameter dictionary
    PARAM_KEYS = ['num_bins', 'la', 'threshold', 'delta']

    def __init__(self, la=2.0, num_bins=128, threshold=1.3, delta=0.03):
        """Constructs and initializes a XiangWMSystem object.

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
        :return None
        """
        self._la = None  # declare and set to none, so that it's not defined
        # outside of the constructor
        self._num_bins = None
        self._threshold = None
        self._delta = None
        self._MIN_THRESHOLD = 1.1
        self.set_params(la=la, num_bins=num_bins, threshold=threshold,
                        delta=delta)

        self._is_init = True

    def set_params(self, **kwargs):
        """Sets the parameters of Xiang's watermarking system. This is for
        convenienc and therefore  that uses **kwargs,
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
        :return: None
        """
        if 'la' in kwargs:
            self.la = kwargs['la']

        if 'num_bins' in kwargs:
            self.num_bins = int(kwargs['num_bins'])

        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']

        if 'delta' in kwargs:
            self.delta = kwargs['delta']

    def get_params(self):
        """Returns all needed parameters as a dictionary

        :return: a dictionary containing all parameters
        """

        return dict(zip(XiangWMSystem.PARAM_KEYS,
                        [self.num_bins, self.la, self.threshold, self.delta]))

    @property
    def la(self):
        return self._la

    @la.setter
    def la(self, value):
        if value < 2.0 or value > 3.0:
            raise ValueError('Lambda exceeds the suggested range')
        self._la = value

    @property
    def num_bins(self):
        return self._num_bins

    @num_bins.setter
    def num_bins(self, value):
        self._num_bins = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if value < self._MIN_THRESHOLD:
            raise ValueError('Threshold too small to resist TSM')
        self._threshold = value

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, value):
        if value < 0:
            raise ValueError('Delta must be > 0')
        self._delta = value

    def embed_watermark(self, samples, w, **kwargs):
        """Embeds the specified mark in the samples.

        :param samples: the signal to be marked
        :param w: the watermark
        :param kwargs: the embedding key (in this case lambda) - optionally
        (cause it is already set)
        :return: None
        """

        assert self._is_init, "WM System not initialized"

        if 'key' in kwargs:
            key = kwargs['key']
            self.la = key

        print('=============================================')
        print("Embedding ", w, " via Xiang's method")
        print('---------------------------------------------')

        # make a deep copy of the samples to mark
        samples_to_mark = np.empty_like(samples)
        samples_to_mark[:] = samples

        if samples.ndim > 1:
            length, num_channels = samples.shape
            wmk = self.check_wmk_alignment(num_channels, w)

            for i in range(0, num_channels):
                print('in channel #', i)
                print('---------------------------------------------')
                self.embed_watermark_single_channel(samples_to_mark[:, i],
                                                    wmk[i])

        else:
            print('in channel #0')
            print('---------------------------------------------')
            self.embed_watermark_single_channel(samples_to_mark, w)

        return samples_to_mark

    def embed_watermark_single_channel(self, samples_to_mark, w):
        """Embeds the watermark in a mono signal.

        :param samples_to_mark: the samples, which are marked
        :param w: the watermark
        :return: None
        """
        assert self._num_bins >= 3 * len(
            w), "Number of bins < 3*the length of the bitstring to embed"

        hist, bins = self.generate_histogram(samples_to_mark, self.la,
                                             self._num_bins,
                                             self.mean_of_absolute_values(
                                                 samples_to_mark))
        bin_width = bins[1] - bins[0]

        for ind, bit in enumerate(w):
            start_bin = ind * 3

            a = hist[start_bin]
            b = hist[start_bin + 1]
            c = hist[start_bin + 2]

            if bit == 1:
                v = 2 * b / (a + c)
                if v >= self.threshold:
                    # no operation is needed
                    continue
                else:
                    i = (self.threshold * (a + c) - 2 * b) / (
                        2 + self.threshold)
                    i1 = (i * a) / (a + c)
                    i3 = (i * c) / (a + c)
                    # adjust number of samples in the bins
                    # take the first i samples, that lie in start_bin and
                    # modify them
                    count = 0
                    for j, x in enumerate(samples_to_mark):
                        if count < i1:
                            if bins[start_bin] <= x < bins[start_bin + 1]:
                                samples_to_mark[j] += bin_width
                                if samples_to_mark[j] < bins[start_bin + 1]:
                                    samples_to_mark[j] += bin_width
                                count += 1
                        else:
                            break

                    count = 0
                    for j, x in enumerate(samples_to_mark):
                        if count < i3:
                            if bins[start_bin + 2] <= x < bins[start_bin + 3]:
                                samples_to_mark[j] -= bin_width
                                if samples_to_mark[j] >= bins[start_bin + 2]:
                                    samples_to_mark[j] -= bin_width
                                count += 1
                        else:
                            break

            if bit == 0:
                v = (a + c) / (2 * b)
                if v >= self.threshold:
                    # no operation is needed
                    continue
                else:
                    i = (2 * self.threshold * b - (a + c)) / (
                        1 + 2 * self.threshold)
                    i1 = (i * a) / (a + c)
                    i3 = (i * c) / (a + c)
                    # adjust number of samples
                    count = 0
                    for j, x in enumerate(samples_to_mark):
                        if count < i1:
                            if bins[start_bin + 1] <= x < bins[start_bin + 2]:
                                samples_to_mark[j] -= bin_width
                                if samples_to_mark[j] >= bins[start_bin + 1]:
                                    samples_to_mark[j] -= bin_width
                                count += 1
                        else:
                            break
                    count = 0
                    for j, x in enumerate(samples_to_mark):
                        if count < i3:
                            if bins[start_bin + 1] <= x < bins[start_bin + 2]:
                                samples_to_mark[j] += bin_width
                                if samples_to_mark[j] < bins[start_bin + 2]:
                                    samples_to_mark[j] += bin_width
                                count += 1
                        else:
                            break

    def extract_watermark(self, samples, **kwargs):
        """Extracts the watermark from the given samples.

        :param samples: the marked samples
        :param kwargs: various parameters; syn is required, la, which is
        kind of the detection key, could be handed
        :return: w: the extracted mark - if samples is a multi-channel
        signal, then a list of marks is returned
        """

        assert self._is_init, 'WM System not initialized'

        if 'syn' in kwargs:
            syn = kwargs['syn']
        else:
            raise TypeError('Synchronization code required')

        if 'key' in kwargs:
            self.la = kwargs['key']

        if 'length' in kwargs:
            wmk_len = kwargs['length']
        else:
            wmk_len = self.num_bins // 3

        print('=============================================')
        print("Detecting watermark")
        print('---------------------------------------------')

        # multi channel signal
        if samples.ndim > 1:
            length, num_channels = samples.shape

            if not isinstance(wmk_len, (list, tuple)):
                wmk_len = [wmk_len] * num_channels

            syn = self.check_wmk_alignment(num_channels, syn)

            w = np.array([])
            for i in range(0, num_channels):
                print('in channel #', i)
                print('---------------------------------------------')

                w_i = self.extract_watermark_single_channel(samples[:, i],
                                                            syn[i])
                w_i = w_i[:wmk_len[i]]  # rip off abundant bits

                # store extracted watermark in 2d output array
                if i == 0:
                    w = np.array(w_i)
                else:
                    w = np.vstack((w, w_i))

        # mono signal
        else:
            print('in channel #0')
            print('---------------------------------------------')

            w = self.extract_watermark_single_channel(samples, syn)
            w = w[:wmk_len]  # rip off abundant bits

        return w

    def extract_watermark_single_channel(self, samples, syn):
        """Extracts the watermark from a marked mono signal.

        :param samples: the marked single channel signal
        :param syn: the synchronization code (part of the watermark)
        :return: w: the extracted mark
        """
        mean = HistBasedWMSystem.mean_of_absolute_values(samples)

        # 1/la -> enlarge and reduce search space on sample level
        search_space = np.arange(mean * (1 - self.delta),
                                 mean * (1 + self.delta), 1 / self.la)

        best_match = 0
        distance = 1.0
        w = []
        # step through the search space and calculate best match
        for i, m in enumerate(search_space):

            hist, bins = HistBasedWMSystem.generate_histogram(samples, self.la,
                                                              self._num_bins,
                                                              search_space[i])

            extracted_syn = []
            for j in range(0, 3 * len(syn)):
                if j % 3 == 0:

                    a = hist[j]
                    b = hist[j + 1]
                    c = hist[j + 2]
                    if ((2 * b) / (a + c)) >= 1:
                        extracted_syn.append(1)
                    else:
                        extracted_syn.append(0)

            # calculates (2*(n_01 + n10))/(n_00+n_11+2*(n_01 + n10))
            current_distance = scipy.spatial.distance.rogerstanimoto(
                extracted_syn, syn)
            if current_distance <= distance:
                if abs(search_space[i] - mean) < abs(
                                search_space[best_match] - mean):
                    distance = current_distance
                    best_match = i
                    w = extracted_syn

        print('---------------------------------------------')
        print("Syn: ", syn)
        print("Best match: ", w)
        print('---------------------------------------------')

        hist, bins = HistBasedWMSystem.generate_histogram(samples, self.la,
                                                          self._num_bins,
                                                          search_space[
                                                              best_match])

        # Extract watermark from best match in search space
        for i in range(3 * len(syn), self._num_bins):
            if i % 3 == 0 and i + 2 < self._num_bins:
                a = hist[i]
                b = hist[i + 1]
                c = hist[i + 2]
                if ((2 * b) / (a + c)) >= 1:
                    w.append(1)
                else:
                    w.append(0)

        return w

