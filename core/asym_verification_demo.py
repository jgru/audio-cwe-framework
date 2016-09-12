__author__ = 'gru'

import soundfile as sf
import numpy as np

from core.asym_verification_protocol.prover import Prover
from core.asym_verification_protocol.verifier import Verifier
from core.asym_verification_protocol.permutation_utils import LogMapKey
from core.audio_cwe import watermarking_utils


"""
Demonstrates the minimum knowledge verification protocol procedure presented
in the associated thesis. Bob can verify the presence of Alice watermark,
which was embedded via the acquired histogram-based watermarking scheme for
audio data.

Note, this demo is meant to be simple and straightforward. For the
sake of brewity and clarity a short watermark and a very short
signal was chosen. Normally the watermark should have a length of at least
approximately 1000 bits, so that the GI-problem becomes hard to solve.
Refer to the iPython notebook 'minimum_knowledge_verification_demo'
('../notebooks/minimum_knowledge_verification_demo.ipynb') for a more
sophisticated example.
"""

# Create the watermark
wmk = watermarking_utils.construct_watermark('HdM')

# Prepare media object...
samples, samplerate = sf.read(
    '../../res/testing/original_test_files/misc/440_sine.wav', dtype=np.int16)
samples = samples[:, 0]

# ...and watermarking parameters
num_bins = len(wmk) * 10
la = 2.5
threshold = 5
seed = 63


# Create the prover
tau = LogMapKey(3.57567, 0.8, 102)
p = Prover(tau, len(wmk))

# Embed watermark
p.emb(samples, wmk, seed, la=la, num_bins=num_bins, threshold=5, step=5,
      delta=0.0)


# Let a verifier perform the verification
v = Verifier(p, num_rounds=10, seed=1234)
success = v.start_verification()

if success:
    print("Successful verification")
else:
    print("Verification failed")
