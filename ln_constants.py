import numpy as np
import pandas as pd
from scipy.constants import c
from scipy.interpolate import InterpolatedUnivariateSpline

"""
Data Columns:
eV
1 / lambda
wavelength
n_o
k_o
n_e
k_e
src

k_0 = 2pi / lambda
a = 2k'' = 2*k_0*K

"""

ln = pd.read_csv(
    "../data/ln_constants.csv",
    delimiter=",",
    skiprows=1,
    dtype=float,
    names=[
        "eV",
        "cm-1",
        "wavelen",
        "n_ord",
        "k_ord",
        "n_ext",
        "k_ext",
        "src",
    ],
)

# Add frequency, propagation constant, and absorption coefficients to table
ln = ln.assign(freq=c / (ln["wavelen"] * 1e-6))
ln = ln.assign(k_0=2 * np.pi / (ln["wavelen"] * 1e-4))
ln = ln.assign(a_ord=2 * ln["k_0"] * ln["k_ord"])
ln = ln.assign(a_ext=2 * ln["k_0"] * ln["k_ext"])
ln = ln.assign(a_ord_db=ln["a_ord"] * 10 / np.log(10))
ln = ln.assign(a_ext_db=ln["a_ext"] * 10 / np.log(10))

# Remove any rows with NA values and order by frequency
ln = ln.dropna()
ln = ln.sort_values("freq")
