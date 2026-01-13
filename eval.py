import cv2
import numpy as np
import csv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# kernels
data = [["kernels"]]
with open("output.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)

for k in range(1, 9):
    temp = np.zeros(4)
    for ind in range(1, 13):
        B = (
            cv2.imread(
                "all_results/k{}/blur_noised/{}.png".format(k, ind),
                cv2.IMREAD_GRAYSCALE,
            ).astype(np.float64)
            / 255
        )
        I_gt = (
            cv2.imread("data/sharp/{}.png".format(ind), cv2.IMREAD_GRAYSCALE).astype(
                np.float64
            )
            / 255
        )
        I_opt = (
            cv2.imread(
                "all_results/k{}/deblur/{}.png".format(k, ind),
                cv2.IMREAD_GRAYSCALE,
            ).astype(np.float64)
            / 255
        )
        assert np.isrealobj(I_opt), "I_opt is complex"
        assert B.dtype == np.float64
        assert I_gt.dtype == np.float64
        assert I_opt.dtype == np.float64
        assert B.min() >= 0.0 and B.max() <= 1.0
        assert I_gt.min() >= 0.0 and I_gt.max() <= 1.0
        assert I_opt.min() >= 0.0 and I_opt.max() <= 1.0

        old_SSIM = ssim(B, I_gt, data_range=1.0)
        new_SSIM = ssim(I_opt, I_gt, data_range=1.0)
        old_PSNR = psnr(B, I_gt, data_range=1.0)
        new_PSNR = psnr(I_opt, I_gt, data_range=1.0)
        temp[0] += old_SSIM
        temp[1] += new_SSIM
        temp[2] += old_PSNR
        temp[3] += new_PSNR
        # excel for recording experiment data
        data = [
            [
                ind,
                old_SSIM,
                new_SSIM,
                old_PSNR,
                new_PSNR,
            ]
        ]
        with open("output.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(data)
    with open("output.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows([["avg"], temp / 12])

# stds
data = [["stds"]]
with open("output.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)

for s in range(1, 4):
    temp = np.zeros(4)
    for ind in range(1, 13):
        B = (
            cv2.imread(
                "all_results/std00{}/blur_noised/{}.png".format(s, ind),
                cv2.IMREAD_GRAYSCALE,
            ).astype(np.float64)
            / 255
        )
        I_gt = (
            cv2.imread("data/sharp/{}.png".format(ind), cv2.IMREAD_GRAYSCALE).astype(
                np.float64
            )
            / 255
        )
        I_opt = (
            cv2.imread(
                "all_results/std00{}/deblur/{}.png".format(s, ind),
                cv2.IMREAD_GRAYSCALE,
            ).astype(np.float64)
            / 255
        )
        assert np.isrealobj(I_opt), "I_opt is complex"
        assert B.dtype == np.float64
        assert I_gt.dtype == np.float64
        assert I_opt.dtype == np.float64
        assert B.min() >= 0.0 and B.max() <= 1.0
        assert I_gt.min() >= 0.0 and I_gt.max() <= 1.0
        assert I_opt.min() >= 0.0 and I_opt.max() <= 1.0

        old_SSIM = ssim(B, I_gt, data_range=1.0)
        new_SSIM = ssim(I_opt, I_gt, data_range=1.0)
        old_PSNR = psnr(B, I_gt, data_range=1.0)
        new_PSNR = psnr(I_opt, I_gt, data_range=1.0)
        temp[0] += old_SSIM
        temp[1] += new_SSIM
        temp[2] += old_PSNR
        temp[3] += new_PSNR
        # excel for recording experiment data
        data = [
            [
                ind,
                old_SSIM,
                new_SSIM,
                old_PSNR,
                new_PSNR,
            ]
        ]
        with open("output.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(data)
    with open("output.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows([["avg"], temp / 12])

# alphas
data = [["alphas"]]
with open("output.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)

for alpha in range(3, 8, 2):
    temp = np.zeros(4)
    for ind in range(1, 13):
        B = (
            cv2.imread(
                "all_results/alpha0{}/blur_noised/{}.png".format(alpha, ind),
                cv2.IMREAD_GRAYSCALE,
            ).astype(np.float64)
            / 255
        )
        I_gt = (
            cv2.imread("data/sharp/{}.png".format(ind), cv2.IMREAD_GRAYSCALE).astype(
                np.float64
            )
            / 255
        )
        I_opt = (
            cv2.imread(
                "all_results/alpha0{}/deblur/{}.png".format(alpha, ind),
                cv2.IMREAD_GRAYSCALE,
            ).astype(np.float64)
            / 255
        )
        assert np.isrealobj(I_opt), "I_opt is complex"
        assert B.dtype == np.float64
        assert I_gt.dtype == np.float64
        assert I_opt.dtype == np.float64
        assert B.min() >= 0.0 and B.max() <= 1.0
        assert I_gt.min() >= 0.0 and I_gt.max() <= 1.0
        assert I_opt.min() >= 0.0 and I_opt.max() <= 1.0

        old_SSIM = ssim(B, I_gt, data_range=1.0)
        new_SSIM = ssim(I_opt, I_gt, data_range=1.0)
        old_PSNR = psnr(B, I_gt, data_range=1.0)
        new_PSNR = psnr(I_opt, I_gt, data_range=1.0)
        temp[0] += old_SSIM
        temp[1] += new_SSIM
        temp[2] += old_PSNR
        temp[3] += new_PSNR
        # excel for recording experiment data
        data = [
            [
                ind,
                old_SSIM,
                new_SSIM,
                old_PSNR,
                new_PSNR,
            ]
        ]
        with open("output.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(data)
    with open("output.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows([["avg"], temp / 12])
