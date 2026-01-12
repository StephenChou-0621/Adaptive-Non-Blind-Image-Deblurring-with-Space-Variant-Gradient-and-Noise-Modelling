import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def read_img(bpath, cpath):
    B = cv2.imread(bpath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    I = cv2.imread(cpath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    return B, I


def show(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def make_noisy_img(img, noise):
    if img.max() > 1:
        img = img.astype(float) / 255
    return np.clip(img + noise, 0, 1)


def x_grad_map(B):
    gx = np.array([[1, 0, -1]], dtype=float)
    return convolve2d(B, gx, mode="same", boundary="symm")


def y_grad_map(B):
    gy = np.array([[1, 0, -1]], dtype=float).reshape(3, 1)
    return convolve2d(B, gy, mode="same", boundary="symm")


def deconv(blurimg, alpha, alpha_n, lam, kernel=None):
    def fast_deconvolution(yin, k, alpha, alpha_n, lam):
        def denominator(y, k):
            def psf2otf(k, size_y):
                K = np.zeros(size_y, dtype=complex)
                h, w = k.shape
                K[:h, :w] = k
                K = np.roll(K, -(h // 2), axis=0)
                K = np.roll(K, -(w // 2), axis=1)
                K = np.fft.fft2(K)
                return K

            M, N = y.shape
            K = psf2otf(k, (M, N))
            Y = np.fft.fft2(y)
            Nomin1 = np.conj(K) * Y  # K^T*B
            Denom1 = np.abs(K) ** 2  # K^2

            gx = np.array([1, -1]).reshape(1, 2)
            gy = np.array([1, -1]).reshape(2, 1)
            Gx = psf2otf(gx, (M, N))
            Gy = psf2otf(gy, (M, N))
            Denom2 = np.abs(Gx) ** 2 + np.abs(Gy) ** 2  # Fx^2+Fy^2
            return Nomin1, Denom1, Denom2, K

        LUT = {}
        rang = 10
        step = 0.0001
        xx = np.arange(-rang, rang + step, step)

        def solve_img(v, beta, alpha):
            def compute_w1(v, beta):
                ans = np.sign(v) * np.max(np.abs(v) - 1 / beta, 0)
                return ans

            def compute_w23(v, beta):
                # Ferrari's method in alg.3 in the NIPS paper
                # f(w)=w^4-3vw^3+3v^2w^2-v^3w+8/(27beta^3)=0, w=?
                eps = 1e-6
                m = np.full_like(v, 8 / (27 * beta**3))  # for same shape
                t1 = (-9 / 8) * v**2
                t2 = (1 / 4) * v**3
                t3 = (-1 / 8) * m * v**2
                t4 = -t3 / 2 + np.sqrt(
                    (-(m**3) / 27 + (m**2 * v**4) / 256).astype(np.complex128)
                )
                t5 = t4 ** (1 / 3)
                t6 = 2 * (-5 / 18 * t1 + t5 + (m / (3 * t5)))
                t7 = np.sqrt((t1 / 3 + t6).astype(np.complex128))

                # roots
                root = np.zeros(v.shape + (4,), dtype=complex)
                root[:, 0] = (3 / 4) * v + (
                    t7 + np.sqrt((-(t1 + t6 + t2 / t7)).astype(np.complex128))
                ) / 2
                root[:, 1] = (3 / 4) * v + (
                    t7 - np.sqrt((-(t1 + t6 + t2 / t7)).astype(np.complex128))
                ) / 2
                root[:, 2] = (3 / 4) * v + (
                    -t7 + np.sqrt((-(t1 + t6 - t2 / t7)).astype(np.complex128))
                ) / 2
                root[:, 3] = (3 / 4) * v + (
                    -t7 - np.sqrt((-(t1 + t6 - t2 / t7)).astype(np.complex128))
                ) / 2

                # choose best root
                c1 = np.abs(np.imag(root)) < eps  # is real root
                vv = v[:, None]
                c23 = np.real(root) * np.sign(vv)  # root direction
                c1 &= (c23 > ((1 / 2) * np.abs(vv))) & (
                    c23 < np.abs(vv)
                )  # 2/3|v|<|r|<|v| and sign(r)=sign(v) -> equ(13)
                root[~c1] = 0
                root = np.real(root)
                ans = np.max(root, axis=1)
                return ans

            def compute_w12(v, beta):
                # f(w)=w^3-2vw^2+v^2w-sign(v)/(4beta^2)=0, w=?
                # Cardano's method in alg.2 of the NIPS paper
                eps = 1e-6
                m = -np.sign(v) / (4 * beta**2)
                t1 = (2 / 3) * v
                t2 = (
                    -27 * m
                    - 2 * v**3
                    + 3 ** (3 / 2)
                    * np.sqrt((27 * m**2 + 4 * m * v**3).astype(np.complex128))
                ) ** (1 / 3)
                t2 = np.where(np.abs(t2) < eps, eps, t2)
                t3 = v**2 / t2

                # roots
                root = np.zeros(v.shape + (3,), dtype=complex)
                root[:, 0] = t1 + (2 ** (1 / 3)) / 3 * t3 + t2 / (3 * 2 ** (1 / 3))
                root[:, 1] = (
                    t1
                    - ((1 + 1j * np.sqrt(3)) / (3 * 2 ** (2 / 3))) * t3
                    - ((1 - 1j * np.sqrt(3)) / (6 * 2 ** (1 / 3))) * t2
                )
                root[:, 2] = (
                    t1
                    - ((1 - 1j * np.sqrt(3)) / (3 * 2 ** (2 / 3))) * t3
                    - ((1 + 1j * np.sqrt(3)) / (6 * 2 ** (1 / 3))) * t2
                )

                # choose best root
                root = np.where(np.isfinite(root), root, 0)  # remove inf
                vv = v[:, None]
                c23 = np.real(root) * np.sign(vv)  # root direction
                c1 = np.abs(np.imag(root)) < eps  # is real root
                c1 &= (c23 > ((2 / 3) * np.abs(vv))) & (
                    c23 < np.abs(vv)
                )  # 2/3|v|<|r|<|v| and sign(r)=sign(v) -> equ(13)
                root[~c1] = 0
                root = np.real(root)
                ans = np.max(root, axis=1)
                return ans

            def newton_w(v, beta, alpha):
                # f(w)=|w|^alpha+(beta/2)*(w-v)^2
                # f'(w)=alpha*|w|^(alpha-1)*sign(w)+beta*(w-v)
                # f''(w)=alpha*(alpha-1)*|w|^(alpha-2)+beta
                # w=w0-f'(w)/f''(w)
                w = v.copy().astype(np.float32)
                for time in range(4):
                    df = alpha * np.sign(w) * np.abs(w) ** (alpha - 1) + beta * (w - v)
                    ddf = alpha * (alpha - 1) * np.abs(w) ** (alpha - 2) + beta
                    w -= df / ddf
                w = np.where(np.isfinite(w), w, 0)  # remove inf

                # to choose w=0 or w=w*
                cost0 = (beta / 2) * v**2
                costw = np.abs(w) ** alpha + (beta / 2) * (w - v) ** 2

                # w=0 if costw>cost0, w=w* if costw<cost0
                ans = np.where(costw < cost0, w, 0)
                return ans

            def compute_w(v, beta, alpha):
                eps = 1e-9
                if abs(alpha - 1) < eps:
                    return compute_w1(v, beta)
                if abs(alpha - 2 / 3) < eps:
                    return compute_w23(v, beta)
                if abs(alpha - 1 / 2) < eps:
                    return compute_w12(v, beta)
                return newton_w(v, beta, alpha)

            key = (beta, alpha)
            if key in LUT:
                w_value = LUT[key]
            else:
                w_value = compute_w(xx, beta, alpha)
                LUT[key] = w_value
            temp = np.interp(v.flatten(), xx, w_value)  # interpolation
            W = temp.reshape(v.shape)
            return W

        yout = yin.copy()
        l = k.shape[0]
        if l % 2 == 0:
            raise ValueError("kernel size must be odd")

        # compute F{x}
        Nomin1, Denom1, Denom2, K = denominator(yin, k)

        # compute F{w}
        youtx = np.roll(yout, -1, axis=1) - yout
        youty = np.roll(yout, -1, axis=0) - yout
        youtn = yin - np.fft.ifft2(np.fft.fft2(yout) * K).real

        # main loop
        betas = np.geomspace(1, 2**8, num=9)
        gamma = 1 / 50  # =beta_g/beta_n
        beta_n = betas * lam / gamma
        beta_g = betas

        for i in range(len(betas)):
            # w-subproblem
            Wn = solve_img(
                youtn, beta_n[i], alpha_n
            )  # *const -> denoise; /const -> deblur
            Wx = solve_img(
                youtx, beta_g[i], alpha
            )  # *const -> deblur; /const -> denoise
            Wy = solve_img(youty, beta_g[i], alpha)

            # x-subproblem
            Wxx = np.roll(Wx, 1, axis=1) - Wx
            Wyy = np.roll(Wy, 1, axis=0) - Wy
            Wnn = np.fft.ifft2(np.fft.fft2(Wn) * K.conj()).real

            W = -Wnn + gamma * (Wxx + Wyy)
            W = np.fft.fft2(W)

            Denom = Denom1 + Denom2 * gamma
            Fyout = (W + Nomin1) / Denom
            yout = np.fft.ifft2(Fyout).real
            yout = np.clip(yout, 0, 1)

            # update youtx and youty (gradient) and youtn
            youtx = np.roll(yout, -1, axis=1) - yout
            youty = np.roll(yout, -1, axis=0) - yout

            youtn = yin - np.fft.ifft2(np.fft.fft2(yout) * K).real
        return yout

    psf = kernel
    k_size = psf.shape[0]
    # mirror-padding the coded img
    M, N = blurimg.shape
    temp = np.zeros((M + 2 * k_size, N + 2 * k_size))
    temp[k_size : k_size + M, k_size : k_size + N] = blurimg
    for m in range(M + 2 * k_size):
        for n in range(N + 2 * k_size):
            if m < k_size or m >= M + k_size or n < k_size or n >= N + k_size:
                mm, nn = 0, 0
                if m < k_size:
                    mm = 2 * k_size - m
                elif m >= M + k_size:
                    mm = 2 * (M + k_size - 1) - m
                else:
                    mm = m
                if n < k_size:
                    nn = 2 * k_size - n
                elif n >= N + k_size:
                    nn = 2 * (N + k_size - 1) - n
                else:
                    nn = n
                temp[m, n] = temp[mm, nn]
    blurimg = temp.copy()

    # deconvolution
    x = fast_deconvolution(blurimg, psf, alpha, alpha_n, lam)

    # remove padding
    x = x[k_size : k_size + M, k_size : k_size + N]

    return x
