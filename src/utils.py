import numpy as np
from math import gamma as Γ


def hyper_Laplacian_noise(shape, alpha, std, seed=None):
    rng = np.random.default_rng(seed)
    beta = std * np.sqrt(Γ(1 / alpha) / Γ(3 / alpha))
    T = rng.gamma(shape=1 / alpha, scale=1.0, size=shape)
    S = rng.choice([-1.0, 1.0], size=shape)
    return beta * S * (T ** (1.0 / alpha))


def Eg(g):
    return np.sum(g * g)


def sigma_noise(sgx_n, sgy_n, E):
    epsilon = 1e-8
    sigma_n = np.sqrt((sgx_n**2 + sgy_n**2 + epsilon) / E)
    return sigma_n


def sigma_gs(sg, sg_n):
    epsilon = 1e-8
    sgs = sg**2 - sg_n**2
    sgs[sgs < epsilon] = epsilon
    return np.sqrt(sgs)


def lambda_library(alpha, C, N):
    i = np.arange(N, dtype=float)  # i = 0 ~ N-1
    lam = C * (2 ** ((alpha / 3) * i))
    return lam
