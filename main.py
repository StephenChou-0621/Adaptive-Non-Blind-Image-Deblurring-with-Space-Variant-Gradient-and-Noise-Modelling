import cv2
import numpy as np
import os
from os.path import join
import sys
import csv
import src.image_ops as ops
import src.param_est as param
import src.utils as utils
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def main():
    # NOTE: you can try other noise level by setting std_noise_set = [0.01,0.05]
    # to set clean image, set std_noise_set = 0
    alpha, std_noise_set = 0.8, 0.01
    _, bpath, cpath, rpath1, rpath2, kpath = sys.argv
    B, I_gt = ops.read_img(bpath, cpath)  # [0,1]
    M, N = B.shape
    kernel = cv2.imread(kpath, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    kernel /= kernel.sum()

    # add noise
    # NOTE: you can try other alpha_n_set in [0.1,0.9]
    alpha_n_set = 0.3
    noise = utils.hyper_Laplacian_noise((M, N), alpha_n_set, std_noise_set, seed=0)
    B = ops.make_noisy_img(B, noise)

    # save noised blur img
    B_int = B * 255
    B_int = B_int.astype(np.uint8)
    os.makedirs(rpath1, exist_ok=True)
    filename = os.path.basename(bpath)
    ind, _ = os.path.splitext(filename)
    cv2.imwrite(join(rpath1, "{}.png".format(int(ind))), B_int)
    temp = join(rpath1, "{}.png".format(int(ind)))
    B = cv2.imread(temp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    # step 1,2
    # std_n
    B_HH = param.DWT_HH(B)
    Bgx, Bgy = ops.x_grad_map(B_HH), ops.y_grad_map(B_HH)
    L = 10
    sgx, sgy = param.local_std(Bgx, L), param.local_std(Bgy, L)

    sort_sgx, sort_sgy = np.sort(sgx.flatten()), np.sort(sgy.flatten())
    sgx_n, sgy_n = (
        param.find_turning_pt(sort_sgx, M, N),
        param.find_turning_pt(sort_sgy, M, N),
    )
    sigma_n = utils.sigma_noise(sgx_n, sgy_n, utils.Eg(np.array([1, 0, -1])))

    # print the estimated noise std
    print("std_n=", sigma_n)

    # sigma_gsx,y
    Bgx, Bgy = ops.x_grad_map(B), ops.y_grad_map(B)
    sgx, sgy = param.local_std(Bgx, L), param.local_std(Bgy, L)
    sigma_gsx, sigma_gsy = utils.sigma_gs(sgx, sgx_n), utils.sigma_gs(sgy, sgy_n)

    # set alpha_noise=alpha_image gradient in first round restoration
    alpha_n = alpha

    # 1D-interpolation deconv.
    # step 3
    lam_map = (np.sqrt(2 * sigma_n**2 / (sigma_gsx**2 + sigma_gsy**2))) ** alpha
    C = lam_map.min()
    lam_N = int(np.ceil(3 / alpha * np.log2(lam_map.max() / C)) + 2)
    lam_library = utils.lambda_library(alpha, C, lam_N)

    # step 4
    I_library = {}
    sat, prev_I = False, None
    for i in range(len(lam_library)):
        if not sat:
            I_library[i] = ops.deconv(B, alpha, alpha_n, lam_library[i], kernel)
        else:
            I_library[i] = prev_I.copy()
            continue
        if np.array_equal(I_library[i], prev_I):
            sat = True
        else:
            prev_I = I_library[i].copy()

    w_map = np.zeros_like(lam_map)
    index_map = np.zeros_like(lam_map)
    M, N = w_map.shape
    for m in range(M):
        for n in range(N):
            i = int(np.ceil((3 / alpha) * np.log2(lam_map[m, n] / C)))
            w_map[m, n] = (
                np.log(lam_library[i + 1] / lam_map[m, n])
                / np.log(lam_library[i + 1] / lam_library[i])
            ) ** 1.4
            index_map[m, n] = i
    I_opt = np.zeros_like(lam_map)
    for m in range(M):
        for n in range(N):
            I_opt[m, n] = (
                w_map[m, n] * I_library[index_map[m, n]][m, n]
                + (1 - w_map[m, n]) * I_library[index_map[m, n] + 1][m, n]
            )
    I_opt = np.clip(I_opt, 0, 1)

    # performance evaluation
    print("without alpha_n estimation")
    old_SSIM = ssim(B, I_gt, data_range=1.0)
    print("old SSIM:", old_SSIM)
    new_SSIM = ssim(I_opt.astype(np.float64), I_gt.astype(np.float64), data_range=1.0)
    print("new SSIM:", new_SSIM)
    old_PSNR = psnr(B, I_gt, data_range=1.0)
    print("old PSNR:", old_PSNR)
    new_PSNR = psnr(I_opt.astype(np.float64), I_gt.astype(np.float64), data_range=1.0)
    print("new PSNR:", new_PSNR)

    # save resulted deblur img
    I_opt = I_opt * 255
    I_opt = I_opt.astype(np.uint8)
    os.makedirs(rpath2, exist_ok=True)
    filename = os.path.basename(bpath)
    ind, _ = os.path.splitext(filename)
    cv2.imwrite(join(rpath2, "{}.png".format(int(ind))), I_opt)
    cv2.imwrite(join(rpath2, "no_exp_est_{}.png".format(int(ind))), I_opt)

    # excel for recording experiment data
    data = [
        [
            ind,
            old_SSIM,
            new_SSIM,
            old_PSNR,
            new_PSNR,
            "alpha_n=",
            alpha_n,
            "alpha_n_set=",
            alpha_n_set,
        ]
    ]
    with open("results/output.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    # deconv again with alpha estimation
    # alpha estimation
    B, I_opt = ops.read_img(
        join(rpath1, "{}.png".format(int(ind))),
        join(rpath2, "{}.png".format(int(ind))),
    )
    alpha_n = param.alpha_estimation(B, I_opt, kernel, sigma_n)

    # print the estimated noise alpha
    print("alpha_n=", alpha_n)

    # deal with failure conditions
    if sigma_n > 0.025 or alpha_n == 0.5:
        alpha_n = max(alpha_n, 0.6)
    if param.kernel_center_value(kernel) < 1e-4:
        alpha_n = 0.8

    #  1D-interpolation deconv.
    I_library = {}
    sat, prev_I = False, None
    for i in range(len(lam_library)):
        if not sat:
            I_library[i] = ops.deconv(B, alpha, alpha_n, lam_library[i], kernel)
        else:
            I_library[i] = prev_I.copy()
            continue
        if np.array_equal(I_library[i], prev_I):
            sat = True
        else:
            prev_I = I_library[i].copy()

    w_map = np.zeros_like(lam_map)
    index_map = np.zeros_like(lam_map)
    M, N = w_map.shape
    for m in range(M):
        for n in range(N):
            i = int(np.ceil((3 / alpha) * np.log2(lam_map[m, n] / C)))
            w_map[m, n] = (
                np.log(lam_library[i + 1] / lam_map[m, n])
                / np.log(lam_library[i + 1] / lam_library[i])
            ) ** 1.4
            index_map[m, n] = i

    I_opt = np.zeros_like(lam_map)
    for m in range(M):
        for n in range(N):
            I_opt[m, n] = (
                w_map[m, n] * I_library[index_map[m, n]][m, n]
                + (1 - w_map[m, n]) * I_library[index_map[m, n] + 1][m, n]
            )
    I_opt = np.clip(I_opt, 0, 1)

    # performance evaluation
    print("with alpha_n estimation")
    old_SSIM = ssim(B, I_gt, data_range=1.0)
    print("old SSIM:", old_SSIM)
    new_SSIM = ssim(I_opt.astype(np.float64), I_gt.astype(np.float64), data_range=1.0)
    print("new SSIM:", new_SSIM)
    old_PSNR = psnr(B, I_gt, data_range=1.0)
    print("old PSNR:", old_PSNR)
    new_PSNR = psnr(I_opt.astype(np.float64), I_gt.astype(np.float64), data_range=1.0)
    print("new PSNR:", new_PSNR)

    # save resulted deblur img
    I_opt = I_opt * 255
    I_opt = I_opt.astype(np.uint8)

    os.makedirs(rpath2, exist_ok=True)
    filename = os.path.basename(bpath)
    ind, _ = os.path.splitext(filename)
    cv2.imwrite(join(rpath2, "{}.png".format(int(ind))), I_opt)

    # excel for recording exp.data
    data = [[ind, old_SSIM, new_SSIM, old_PSNR, new_PSNR, "alpha_n_est=", alpha_n]]
    with open("results/output.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == "__main__":
    main()
