#!/usr/bin/env python

"""Tests for `psfr` package."""

from psfr import psfr
from lenstronomy.Util import kernel_util
import numpy.testing as npt
import numpy as np
from lenstronomy.Util import util
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import astropy.io.fits as pyfits
np.random.seed(42)


def test_shift_psf():
    oversampling = 4
    x, y = 0.2, -0.3
    shift = [x, y]

    from lenstronomy.LightModel.light_model import LightModel
    gauss = LightModel(['GAUSSIAN'])
    numpix = 21
    num_pix_super = numpix * oversampling
    oversampling = 4
    if oversampling % 2 == 0:
        num_pix_super += 1
    sigma = 2
    kwargs_true = [{'amp': 1, 'sigma': sigma, 'center_x': 0, 'center_y': 0}]
    kwargs_shifted = [{'amp': 1, 'sigma': sigma, 'center_x': x, 'center_y': y}]
    x_grid_super, y_grid_super = util.make_grid(numPix=num_pix_super, deltapix=1. / oversampling,
                                                left_lower=False)
    flux_true_super = gauss.surface_brightness(x_grid_super, y_grid_super, kwargs_true)
    psf_true_super = util.array2image(flux_true_super)
    psf_true_super /= np.sum(psf_true_super)

    psf_shifted_super_true = gauss.surface_brightness(x_grid_super, y_grid_super, kwargs_shifted)
    psf_shifted_super_true = util.array2image(psf_shifted_super_true)
    psf_shifted_true = kernel_util.degrade_kernel(psf_shifted_super_true, degrading_factor=oversampling)
    psf_shifted_true = kernel_util.cut_psf(psf_shifted_true, numpix)

    psf_shifted_psfr = psfr.shift_psf(psf_true_super, oversampling, shift, degrade=True, n_pix_star=numpix)

    if False:
        f, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
        vmin, vmax = -5, -1
        ax = axes[0]
        im = ax.imshow(psf_shifted_true - psf_shifted_psfr, origin='lower')
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title('True - Interpol shifted')
        ax = axes[1]
        im = ax.imshow(np.log10(psf_shifted_psfr), origin='lower', vmin=vmin, vmax=vmax)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title('PSF-r numerical shifted')
        plt.show()

    print(np.sum(np.abs(psf_shifted_true - psf_shifted_psfr)), 'sum of absolute residuals')
    npt.assert_almost_equal(psf_shifted_true, psf_shifted_psfr, decimal=5)


def test_linear_amplitude():
    amp = 2
    data = np.ones((5, 5)) * amp
    model = np.ones((5, 5))

    amp_return = psfr._linear_amplitude(data, model)
    npt.assert_almost_equal(amp_return, amp)

    mask = np.ones_like(data)

    amp_return = psfr._linear_amplitude(data, model, mask=mask)
    npt.assert_almost_equal(amp_return, amp)


def test_fit_centroid():
    from lenstronomy.LightModel.light_model import LightModel
    numpix = 41

    x_grid, y_grid = util.make_grid(numPix=numpix, deltapix=1)
    gauss = LightModel(['GAUSSIAN'])
    x_c, y_c = -3.5, 2.2
    kwargs_true = [{'amp': 2, 'sigma': 3, 'center_x': x_c, 'center_y': y_c}]
    kwargs_model = [{'amp': 1, 'sigma': 3, 'center_x': 0, 'center_y': 0}]
    flux_true = gauss.surface_brightness(x_grid, y_grid, kwargs_true)
    flux_true = util.array2image(flux_true)

    flux_model = gauss.surface_brightness(x_grid, y_grid, kwargs_model)
    flux_model = util.array2image(flux_model)

    mask = np.ones_like(flux_true)

    center = psfr.centroid_fit(flux_true, flux_model, mask=mask, variance=None)
    npt.assert_almost_equal(center[0], x_c, decimal=3)
    npt.assert_almost_equal(center[1], y_c, decimal=3)

    variance = np.ones_like(flux_true)
    center = psfr.centroid_fit(flux_true, flux_model, mask=None, variance=variance)
    npt.assert_almost_equal(center[0], x_c, decimal=3)
    npt.assert_almost_equal(center[1], y_c, decimal=3)


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

    # de-shifting in oversampled space (should be a bit lower quality but still better than initial guess)
    psf_after_super = psfr.one_step_psf_estimate(star_list, psf_guess_super, center_list, mask_list=None,
                                                 error_map_list=None, step_factor=0.2, oversampling=oversampling,
                                                 oversampled_residual_deshifting=True)
    diff_after = np.sum((psf_after_super - psf_true_super) ** 2)
    diff_before = np.sum((psf_guess_super - psf_true_super) ** 2)
    assert diff_after < diff_before


def test_saturation_limit():
    # check if psf with saturation limit is more accurate than one without
    import lenstronomy.Util.kernel_util as util
    module_path = os.path.dirname(psfr.__file__)
    psf_filename = module_path + '/Data/JWST_mock/psf_f090w_supersample5_crop.fits'
    kernel = pyfits.getdata(psf_filename)

    oversampling = 5
    saturation_limit = 50
    star_list_webb = []
    x_shift, y_shift = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
    # very bright star added to list of stars and all flux values above saturation limit fixed
    bright_star = psfr.shift_psf(psf_center=kernel, oversampling=5, shift=[x_shift, y_shift], degrade=True, n_pix_star=kernel.shape[0]/oversampling) * 4000
    bright_star[bright_star > saturation_limit] = saturation_limit

    # 5 less bright stars are added
    brightnesses = np.abs(np.random.normal(400 ,100 , 10))
    star_list_webb.append(bright_star)
    for i in range(10):
        x_shift, y_shift = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        # star generated and flux multiplied by relevant brightness factor
        star = psfr.shift_psf(psf_center=kernel, oversampling=5, shift=[x_shift, y_shift], degrade=True, n_pix_star=kernel.shape[0]/oversampling) * brightnesses[i]
        star_list_webb.append(star)

    # psf reconstructed with a saturation limit
    psf_psfr_super_sat, center_list_psfr_super_sat, mask_list_sat = psfr.stack_psf(star_list_webb, oversampling=oversampling, 
                                                  saturation_limit=saturation_limit, num_iteration=10, 
                                                  n_recenter=20)
    # psf reconstructed without a saturation limit
    psf_psfr_super, center_list_psfr_super, mask_list = psfr.stack_psf(star_list_webb, oversampling=oversampling, 
                                                  saturation_limit=None, num_iteration=10, 
                                                  n_recenter=20)
                                                
    kernel_degraded = util.degrade_kernel(kernel, oversampling)
    stacked_psf_sat_degraded = psfr.oversampled2regular(psf_psfr_super_sat, oversampling)
    stacked_psf_degraded = psfr.oversampled2regular(psf_psfr_super, oversampling)

    diff1 = np.sum((stacked_psf_sat_degraded - kernel_degraded)**2)
    diff2 = np.sum((stacked_psf_degraded - kernel_degraded)**2)
    # reconstructed psf with saturation limit should perform better than without
    npt.assert_array_less(diff2, diff1, err_msg='reconstructed psf with saturation limit is worse than without limit')


def test_noisy_psf():
    # create 2 psfs with noisy and noiseless stars. checks if noisy psf has larger residual with the true psf
    import lenstronomy.Util.kernel_util as util
    import lenstronomy.Util.image_util as image_util
    module_path = os.path.dirname(psfr.__file__)
    psf_filename = module_path + '/Data/JWST_mock/psf_f090w_supersample5_crop.fits'
    kernel = pyfits.getdata(psf_filename)

    oversampling = 5
    star_list_webb_noisy = []
    star_list_webb = []
    brightnesses = abs(np.random.normal(loc=600, scale=200, size=(5,)))
    for i in range(5):
        x_shift, y_shift = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        star = psfr.shift_psf(psf_center=kernel, oversampling=5, shift=[x_shift, y_shift], degrade=True, n_pix_star=kernel.shape[0]/oversampling) * brightnesses[i]
        star_list_webb.append(star)
        star_n1 = image_util.add_poisson(star, exp_time = 100.)
        star_n2 = image_util.add_background(star, sigma_bkd = 0.5)
        star_noisy = star + star_n1 + star_n2
        star_list_webb_noisy.append(star_noisy)

    psf_psfr_super_noisy, center_list_psfr_super_sat, mask_list_sat = psfr.stack_psf(star_list_webb_noisy, oversampling=oversampling, 
                                                  saturation_limit=None, num_iteration=10, 
                                                  n_recenter=5)
    psf_psfr_super, center_list_psfr_super_sat, mask_list_sat = psfr.stack_psf(star_list_webb, oversampling=oversampling, 
                                                  saturation_limit=None, num_iteration=10, 
                                                  n_recenter=20)
                                                
    kernel_degraded = util.degrade_kernel(kernel, oversampling)
    stacked_psf_noisy_degraded = psfr.oversampled2regular(psf_psfr_super_noisy, oversampling)
    stacked_psf_degraded = psfr.oversampled2regular(psf_psfr_super, oversampling)

    diff1 = np.sum((stacked_psf_noisy_degraded - kernel_degraded)**2)
    diff2 = np.sum((stacked_psf_degraded - kernel_degraded)**2)
    npt.assert_array_less(diff2, diff1, err_msg='reconstructed psf with noisy stars is better than noiseless stars')


def test_combine_psf():
    from psfr.psfr import combine_psf
    nx, ny = 11, 11
    module_path = os.path.dirname(psfr.__file__)
    psf_filename = module_path + '/Data/JWST_mock/psf_f090w_supersample5_crop.fits'
    kernel = pyfits.getdata(psf_filename)
    nx, ny = np.shape(kernel)
    kernel_list_input = []
    weight_list = np.ones(10)
    for i in range(10):
        kernel_list_input.append(np.random.randn(nx, ny) + kernel)
    diff_input = np.sum((kernel_list_input[0] - kernel) ** 2)

    kernel_new = combine_psf(kernel_list_input, kernel, mask_list=None, weight_list=weight_list, factor=1.,
                             stacking_option='median', symmetry=1, combine_with_old=True)
    diff_output = np.sum((kernel_new - kernel) ** 2)
    assert diff_input > diff_output

    kernel_new = combine_psf(kernel_list_input, kernel, mask_list=None, weight_list=weight_list, factor=1.,
                             stacking_option='median_weight', symmetry=1, combine_with_old=False)
    diff_output = np.sum((kernel_new - kernel) ** 2)
    assert diff_input > diff_output

    kernel_new = combine_psf(kernel_list_input, kernel, mask_list=None, weight_list=None, factor=1.,
                             stacking_option='mean', symmetry=1, combine_with_old=False)
    diff_output = np.sum((kernel_new - kernel) ** 2)
    assert diff_input > diff_output

    npt.assert_raises(ValueError, combine_psf, kernel_list_input, kernel, stacking_option='BAD')
