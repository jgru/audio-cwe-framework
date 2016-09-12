__author__ = 'gru'

import numpy as np

def resample(samples, factor):
    """
    Multiplies the playback speed of a sound speed by a `factor`. This is accomplished by removing samples
    at certain indices or duplicating them. Therefore the pitch is altered also.

    Src: http://zulko.github.io/blog/2014/03/29/soundstretching-and-pitch-shifting-in-python/
    :param samples: the samples to resample
    :param factor: the 'scaling' factor
    :return resampled samples
    """
    ids = np.round(np.arange(0, len(samples), factor))
    ids = ids[ids < len(samples)].astype(int)

    return samples[ids]
