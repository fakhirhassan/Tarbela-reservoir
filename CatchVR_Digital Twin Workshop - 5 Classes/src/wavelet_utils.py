
import numpy as np
import pywt

def wavelet_denoise(x, wavelet="db5", level=1):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs_d = [pywt.threshold(c, thr, mode="soft") if i > 0 else c for i, c in enumerate(coeffs)]
    return pywt.waverec(coeffs_d, wavelet)

