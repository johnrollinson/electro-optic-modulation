from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def n(x, a, b, c):
    try:
        return np.sqrt(a + b * np.square(x) / (np.square(x) - c))
    except FloatingPointError:
        return np.nan
    except RuntimeWarning as m:
        print(x, a, b, c)
        return np.nan


def range_by_value(arr, start, stop):
    start_i = np.where(arr > start)[0][0]
    stop_i = np.where(arr < stop)[0][-1]
    return start_i, stop_i


pfile = ["../data/n_opt_param_o.csv", "../data/n_opt_param_e.csv"]

# If the data has already been fit, simply load the fit parameters
# Otherwise, perform the fitting and save the parameters
if path.exists(pfile[0]) and path.exists(pfile[1]):
    param_o = np.loadtxt(pfile[0]).T
    param_e = np.loadtxt(pfile[1]).T
else:
    data = np.loadtxt("../data/n_vs_l_opt.csv", delimiter="\t", skiprows=1).T

    # Extract data from text file
    lam = data[0]
    n_o_l = data[1]
    n_e_l = data[2]

    # Perform the curve fitting
    i, j = range_by_value(lam, 0.25, 3.0)
    param_o, _ = curve_fit(n, lam[i:j], n_o_l[i:j], p0=(1, 1, 0.01))
    param_e, _ = curve_fit(n, lam[i:j], n_e_l[i:j], p0=(1, 1, 0.01))

    # Save the fit params
    np.savetxt(pfile[0], param_o)
    np.savetxt(pfile[1], param_e)


def n_o(wavelen):
    return n(wavelen, *param_o)


def n_e(wavelen):
    return n(wavelen, *param_e)
