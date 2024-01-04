import numpy as np


# def modulation(f, t):
#     return np.sin(2 * np.pi * f * t)


# def carrier(f, t, mod_index, phi, phi_0):
#     return np.sin(2 * np.pi * f * t - mod_index * phi - phi_0)


def ideal_mzi(gamma):
    return 97.02 * np.square(np.cos(gamma / 2)) + 0.98


def ideal_mzi_v(v_mod, v_pi, gamma_0=0):
    gamma = gamma_0 + v_mod * np.pi / v_pi
    return ideal_mzi(gamma)
