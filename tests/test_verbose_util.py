from psfr.verbose_util import verbose_one_step
from psfr import util
import matplotlib.pyplot as plt
import numpy as np


def test_verbose_one_step():
    star_list_jwst = util.jwst_example_stars()
    star = star_list_jwst[0]
    star[star < 0.00001] = 0.00001
    psf_shifted_data = star
    psf_shifted = star
    nx, ny = np.shape(star)
    residuals = np.random.randn(nx, ny)
    residuals_shifted= residuals
    correction = residuals
    psf_new = star
    fig = verbose_one_step(star, psf_shifted, psf_shifted_data, residuals, residuals_shifted, correction, psf_new)
    plt.close()

