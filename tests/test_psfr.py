#!/usr/bin/env python

"""Tests for `psfr` package."""

from psfr import psfr

import numpy.testing as npt
import numpy as np
from lenstronomy.Util import util
import matplotlib.pyplot as plt


def test_linear_amplitude():
    amp = 2
    data = np.ones((5, 5)) * amp
    model = np.ones((5, 5))

    amp_return = psfr._linear_amplitude(data, model)
    npt.assert_almost_equal(amp_return, amp)

    mask = np.ones_like(data)

    amp_return = psfr._linear_amplitude(data, model, mask=mask)
    npt.assert_almost_equal(amp_return, amp)


test_linear_amplitude()


def test_fit_centroid():
    from lenstronomy.LightModel.light_model import LightModel
    numpix = 21
    n_c = (numpix - 1) / 2
    x_grid, y_grid = util.make_grid(numPix=21, deltapix=1, left_lower=True)
    gauss = LightModel(['GAUSSIAN'])
    x_c, y_c = -0.6, 0.2
    kwargs_true = [{'amp': 2, 'sigma': 1, 'center_x': n_c + x_c, 'center_y': n_c + y_c}]
    kwargs_model = [{'amp': 1, 'sigma': 1, 'center_x': n_c, 'center_y': n_c}]
    flux_true = gauss.surface_brightness(x_grid, y_grid, kwargs_true)
    flux_true = util.array2image(flux_true)

    flux_model = gauss.surface_brightness(x_grid, y_grid, kwargs_model)
    flux_model = util.array2image(flux_model)

    mask = np.ones_like(flux_true)

    center = psfr._fit_centroid(flux_true, flux_model, mask=mask, variance=None)
    npt.assert_almost_equal(center[0], x_c, decimal=1)
    npt.assert_almost_equal(center[1], y_c, decimal=1)


def test_one_step_psf_estimation():
    from lenstronomy.LightModel.light_model import LightModel
    numpix = 21
    n_c = (numpix - 1) / 2
    x_grid, y_grid = util.make_grid(numPix=21, deltapix=1, left_lower=True)
    gauss = LightModel(['GAUSSIAN'])
    x_c, y_c = -0.6, 0.2
    sigma = 1
    kwargs_true = [{'amp': 1, 'sigma': sigma, 'center_x': n_c, 'center_y': n_c}]
    flux_true = gauss.surface_brightness(x_grid, y_grid, kwargs_true)
    psf_true = util.array2image(flux_true)
    psf_true /= np.sum(psf_true)

    kwargs_guess = [{'amp': 1, 'sigma': 1.2, 'center_x': n_c, 'center_y': n_c}]
    flux_guess = gauss.surface_brightness(x_grid, y_grid, kwargs_guess)
    psf_guess = util.array2image(flux_guess)
    psf_guess /= np.sum(psf_guess)

    center_list = []
    star_list = []
    displacement_scale = 1
    for i in range(4):
        x_c, y_c = np.random.uniform(-0.5, 0.5) * displacement_scale, np.random.uniform(-0.5, 0.5) * displacement_scale
        center_list.append(np.array([x_c, y_c]))
        kwargs_model = [{'amp': 1, 'sigma': sigma, 'center_x': n_c + x_c, 'center_y': n_c + y_c}]
        flux_model = gauss.surface_brightness(x_grid, y_grid, kwargs_model)
        star = util.array2image(flux_model)
        star_list.append(star)

    psf_after = psfr.one_step_psf_estimate(star_list, psf_guess, center_list, mask_list=None, error_map_list=None,
                                      step_factor=0.2)
    # psf_after should be a better guess of psf_true than psf_guess
    diff_after = np.sum((psf_after - psf_true) ** 2)
    diff_before = np.sum((psf_guess - psf_true) ** 2)

    plt.imshow(psf_true - psf_after)
    plt.colorbar()
    plt.title('after')
    plt.close()
    plt.imshow(psf_true - psf_guess)
    plt.colorbar()
    plt.title('before')
    plt.close()

    plt.imshow(psf_after - psf_guess)
    plt.colorbar()
    plt.title('after - before')
    plt.close()

    print(np.sum(psf_after), np.sum(psf_true))

    assert diff_after < diff_before

    oversampling = 2
    numpix_super = numpix * oversampling
    if oversampling % 2 == 0:
        numpix_super -= 1

    x_grid_super, y_grid_super = util.make_grid(numPix=numpix_super, deltapix=1. / oversampling, left_lower=True)
    flux_guess_super = gauss.surface_brightness(x_grid_super, y_grid_super, kwargs_guess)
    psf_guess_super = util.array2image(flux_guess_super)
    psf_guess_super /= np.sum(psf_guess_super)

    flux_true_super = gauss.surface_brightness(x_grid_super, y_grid_super, kwargs_true)
    psf_true_super = util.array2image(flux_true_super)
    psf_true_super /= np.sum(psf_true_super)

    psf_after_super = psfr.one_step_psf_estimate(star_list, psf_guess_super, center_list, mask_list=None,
                                            error_map_list=None, step_factor=0.2, oversampling=oversampling)
    diff_after = np.sum((psf_after_super - psf_true_super) ** 2)
    diff_before = np.sum((psf_guess_super - psf_true_super) ** 2)
    assert diff_after < diff_before

