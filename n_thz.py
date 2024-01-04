import pickle
from os import path

import numpy as np
from scipy.constants import c
from scipy.interpolate import UnivariateSpline

pfile = ["../data/n_thz_interp_o.pkl", "../data/n_thz_interp_e.pkl"]

# If the data has already been interpolated, simply load the models
# otherwise, perform the interpolation and save the models
if path.exists(pfile[0]) and path.exists(pfile[1]):
    with open(pfile[0], "rb") as n_o_file:
        n_o_interp = pickle.load(n_o_file)
    with open(pfile[1], "rb") as n_e_file:
        n_e_interp = pickle.load(n_e_file)
else:
    data = np.loadtxt("../data/n_vs_l_thz.csv", delimiter="\t", skiprows=1).T

    lam = data[0]
    fm = c / np.flip(lam) * 1e-6
    n_o_l = np.flip(data[1])
    n_e_l = np.flip(data[2])

    n_o_interp = UnivariateSpline(fm, n_o_l, k=2, s=0, ext="const")
    n_e_interp = UnivariateSpline(fm, n_e_l, k=2, s=0, ext="const")

    with open(pfile[0], "wb") as n_o_file:
        pickle.dump(n_o_interp, n_o_file)
    with open(pfile[1], "wb") as n_e_file:
        pickle.dump(n_e_interp, n_e_file)


def n_o(f_thz):
    return n_o_interp(f_thz)


def n_e(f_thz):
    return n_e_interp(f_thz)
