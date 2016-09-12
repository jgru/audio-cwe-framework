__author__ = 'gru'
from core.audio_cwe import watermarking_utils
import soundfile as sf
import numpy as np

b1 = watermarking_utils.construct_watermark(['T', 'E'])
b2 = watermarking_utils.construct_watermark(['T', 'e'])

#print(watermarking_utils.calc_bit_error_rate(b1,b2))
watermarking_utils.compare_watermarks(b1,b2)

b1 = watermarking_utils.construct_watermark('T')
b2 = watermarking_utils.construct_watermark('t')
#print(watermarking_utils.calc_bit_error_rate(b1,b2))
watermarking_utils.compare_watermarks(b1,b2)
