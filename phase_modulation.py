"""
Coherence length, modulation index, and sideband calculations for an x-cut LN
electro-optic phase modulator. Solutions assume E-field applied in the Z
direction, continuous wave laser beam, no reflection or absorption losses,
ignoring geometric considerations (such as waveguide considerations)
"""

import numpy as np
from scipy.constants import c
from scipy.fftpack import fft, fftfreq
from scipy.signal.windows import blackman

import n_opt
import n_thz

# General Parameters
r_33 = 30.8e-12  # [m/V]

# Optical beam
l_opt = 1.55  # [um]


def mod_index(f_thz, E_z, L):
    """
    Calculates the electro-optic-modulation index in terms for x-cut lithium
    niobate with the electro-optic-modulation field applied in the z-direction
    (optic axis). f_thz is the frequency in terahertz, E_z is the electric
    field in V/m, L is the interaction length in mm
    """
    B = np.pi / l_opt * np.power(n_opt.n_e(l_opt), 3) * r_33 * E_z * 1e6
    D = np.pi * f_thz / c * np.abs(n_thz.n_e(f_thz) - n_opt.n_e(l_opt)) * 1e12
    return B * np.sin(D * L * 1e-3) / D


def coherence_length(f_thz, l):
    """
    Returns the coherence length in mm for the x-cut case, where f_thz
    is the electro-optic-modulation frequency in THz, l is the wavelength of
    the laser in um
    """
    return c / (2 * f_thz * np.abs(n_thz.n_o(f_thz) - n_opt.n_e(l))) * 1e-9


def modulation(f, t):
    return np.sin(2 * np.pi * f * t)


def carrier(f, t, mod_index, phi, phi_0=0):
    """Carrier (optical) signal, with phase modulation
    :param f:
    :param t:
    :param mod_index:
    :param phi:
    :return: Phase modulated signal
    """
    return np.sin(2 * np.pi * f * t - mod_index * phi - phi_0)


def spectrum(signal, N, T):
    """Compute the normalized sideband spectrum of the given signal using FFT with
    Blackman windowing.
    """
    window = blackman(N)
    signal = signal * window
    yf = fft(signal)
    xf = fftfreq(N, T)

    yf_norm = yf / max(yf)

    return xf, yf_norm
