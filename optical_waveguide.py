import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

import n_opt

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

"""
Calculations for LN waveguide to determine number of waveguide modes, mode 
field width, and group velocity, assuming a slab waveguide
n1: refractive index of core
n2: refractive index of cladding
2a: slab thickness
"""


def n_sio2(wavelen):
    c1 = (
        0.6961663
        * np.square(wavelen)
        / (np.square(wavelen) - np.square(0.0684043))
    )
    c2 = (
        0.4079426
        * np.square(wavelen)
        / (np.square(wavelen) - np.square(0.1162414))
    )
    c3 = (
        0.8974794
        * np.square(wavelen)
        / (np.square(wavelen) - np.square(9.896161))
    )
    return np.sqrt(c1 + c2 + c3 + 1)


def mode_condition_rhs(theta, wavelen):
    n1 = n_opt.n_e(wavelen)
    n2 = n_sio2(wavelen)
    num = np.sqrt(np.square(np.sin(theta)) - np.square(n2 / n1))
    den = np.cos(theta)
    return num / den


def mode_condition_lhs(theta, wavelen, a, m):
    n1 = n_opt.n_e(wavelen)
    k1 = 2 * np.pi * n1 / wavelen
    lhs = np.tan(a * k1 * np.cos(theta) - m * np.pi / 2)
    lhs[abs(lhs) > 100] = np.nan  # limit range of possible output values
    return lhs


def mode_angle(a, wlen=1.55, plotting=False):
    """
    Solve for the incident angles of all possible waveguide modes
    :param a: waveguide thickness divided by 2 [microns]
    :param wlen: wavelength in microns
    :param plotting: turn plotting of mode equation on/off
    :return:
    """
    theta_m = np.linspace(0, 89.99999, int(5e5)) * np.pi / 180
    rhs = mode_condition_rhs(theta_m, wlen)
    idxx = np.array([], dtype=int)
    for m, l in zip(range(2), ["even", "odd"]):
        lhs = mode_condition_lhs(theta_m, wlen, a, m)
        # Find intersection points
        idx = np.sign(rhs - lhs)  # Convert to sign array
        idx = np.nan_to_num(idx)  # Replace nan values with 0
        idx = np.argwhere(
            np.diff(idx)
        ).flatten()  # Find indices of sign changes
        idx = np.delete(
            idx, [0]
        )  # Remove first element (false sign change from nan replacement)
        idxx = np.append(idxx, idx)
        if plotting:
            plt.plot(theta_m * 180 / np.pi, lhs, label=l)

    idxx = np.sort(idxx)
    idxx = np.delete(idxx, [idxx.size - 1])
    # Remove false values from phase reversal sign change
    for i in range(idxx.size - 1):
        err = np.abs((idxx[i] - idxx[i + 1]) / idxx[i])
        if err < 0.02:
            idxx[i] = 0
            idxx[i + 1] = 0
    idxx = np.delete(idxx, np.argwhere(idxx == 0))
    idxx = idxx[::-1]
    ang = theta_m[idxx]

    if plotting:
        plt.plot(ang * 180 / np.pi, rhs[idxx], "ro")
        plt.plot(theta_m * 180 / np.pi, rhs, label="RHS")

    return ang


def beta(wavelen, thet):
    n1 = n_opt.n_e(wavelen)
    k1 = 2 * np.pi * n1 / wavelen
    return k1 * np.sin(thet)


