import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
from src.utils import hyper_Laplacian_noise
from scipy.stats import entropy
import pywt


def alpha_estimation(B, I_opt, kernel, sigma_n):
    noise_observed = B - I_opt
    B_clean = convolve2d(I_opt, kernel, "same")

    # NOTE: threshold must be < 0.5; otherwise [threshold, 1-threshold] is empty.
    threshold = 0.3

    best_KL = [-0.1, np.inf]
    for i in range(1, 10, 1):
        alpha_n = np.round(0.1 * i, 2)
        noise_ref = hyper_Laplacian_noise(B_clean.shape, alpha_n, sigma_n, seed=0)

        # determine the ther to avoid clip in sample data
        mask = (I_opt <= (1 - threshold)) & (I_opt >= (0 + threshold))

        # faltten and sampling
        noise_sample = noise_observed[mask]
        noise_ref = noise_ref[mask]

        # PDF
        p = 2
        dx = np.round(0.1**p, p)
        noise_sample = np.round(noise_sample, p)
        noise_ref = np.round(noise_ref, p)

        # histogram
        # the valid noise components
        bins = np.arange(-threshold, threshold + dx, dx)
        # ground true -> P
        hist_sample, _ = np.histogram(noise_sample, bins)
        hist_sample = hist_sample.astype(np.float64) / hist_sample.sum()
        hist_sample += 1e-12
        # simulate -> Q
        hist_ref, _ = np.histogram(noise_ref, bins)
        hist_ref = hist_ref.astype(np.float64) / hist_ref.sum()
        hist_ref += 1e-12

        # KL divergence
        kl = entropy(hist_sample, hist_ref)
        if kl < best_KL[1]:
            best_KL = [alpha_n, kl]
    return best_KL[0]


def DWT_HH(img):
    LL, (LH, HL, HH) = pywt.dwt2(img, "db2")
    return HH


def kernel_center_value(kernel):
    h = np.array(kernel, dtype=np.float64)
    h /= h.sum()
    cx, cy = np.array(h.shape) // 2
    center_val = h[cx, cy]
    return center_val


def local_std(Bg, L):
    # assume mean of Bg is zero
    win_size = 2 * L + 1
    k = np.ones((win_size, win_size), dtype=np.float64) / (win_size**2)
    ms_local = convolve2d(Bg**2, k, mode="same", boundary="symm")
    return np.sqrt(ms_local)


def find_turning_pt(sort_g_std, M, N):
    original_sg = sort_g_std.copy()

    # filter
    d = int(M * N / 2000)
    smooth_filter = np.ones(d * 2 + 1)
    smooth_filter[0] = 0
    smooth_filter[d + 1 :] = -1
    sort_g_std = convolve1d(sort_g_std, smooth_filter, mode="reflect")

    # find the first turning point -> the first local maximum of the filtered sort_g_std
    ind = 0
    for i in range(1, len(sort_g_std)):
        if sort_g_std[i] < sort_g_std[ind]:
            break
        else:
            ind = i - 1
    return original_sg[ind]

