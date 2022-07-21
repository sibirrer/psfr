from psfr.psfr import stack_psf
from psfr.util import oversampled2regular
from lenstronomy.Util import kernel_util
from lenstronomy.Util import util
import numpy as np


def stack_psf_guassian_high_res(oversampling, num_stars, kwargs_one_step, num_iteration, n_recenter):
    """"
    """
    from lenstronomy.LightModel.light_model import LightModel
    numpix = 21
    oversampling_compute = 5

    numpix_super = numpix * oversampling * oversampling_compute
    if numpix_super % 2 == 0:
        numpix_super -= 1

    numpix_grid = numpix * oversampling_compute
    if numpix_grid % 2 == 0:
        numpix_grid -= 1

    x_grid, y_grid = util.make_grid(numPix=numpix_grid, deltapix=1. / oversampling_compute, left_lower=False)
    x_grid_super, y_grid_super = util.make_grid(numPix=numpix_super, deltapix=1. / oversampling / oversampling_compute,
                                                left_lower=False)
    gauss = LightModel(['GAUSSIAN'])
    sigma = 1
    kwargs_true = [{'amp': 1, 'sigma': sigma, 'center_x': 0, 'center_y': 0}]
    flux_true = gauss.surface_brightness(x_grid_super, y_grid_super, kwargs_true)
    psf_true = util.array2image(flux_true)
    psf_true = oversampled2regular(psf_true, oversampling=oversampling_compute)
    psf_true /= np.sum(psf_true)

    kwargs_guess = [{'amp': 1, 'sigma': 1.5, 'center_x': 0, 'center_y': 0}]
    flux_guess = gauss.surface_brightness(x_grid_super, y_grid_super, kwargs_guess)
    psf_guess = util.array2image(flux_guess)
    psf_guess = oversampled2regular(psf_guess, oversampling=oversampling_compute)
    psf_guess /= np.sum(psf_guess)

    center_list = []
    star_list = []
    scatter_scale = 1
    for i in range(num_stars):
        x_c, y_c = np.random.uniform(-0.5, 0.5) * scatter_scale, np.random.uniform(-0.5, 0.5) * scatter_scale
        center_list.append(np.array([x_c, y_c]))
        amp = np.random.uniform([0.1, 10])
        kwargs_model = [{'amp': 1, 'sigma': sigma, 'center_x': x_c, 'center_y': y_c}]
        flux_model = gauss.surface_brightness(x_grid, y_grid, kwargs_model)
        star = util.array2image(flux_model)
        star = oversampled2regular(star, oversampling=oversampling_compute)
        star_list.append(star)

    psf_after, center_list_after, mask_list = stack_psf(star_list, oversampling=oversampling,
                                                        saturation_limit=None, num_iteration=num_iteration,
                                                        n_recenter=n_recenter, verbose=False, kwargs_one_step=kwargs_one_step)

    psf_true = kernel_util.cut_psf(psf_true, len(psf_after))
    psf_guess = kernel_util.cut_psf(psf_guess, len(psf_after))
    assert np.sum((psf_after - psf_true) ** 2) < np.sum((psf_guess - psf_true) ** 2)


def test_gaussian():
    num_stars = 20
    num_iterations = 20
    n_recenter = 50

    deshift_order_list = [1]  # [0, 1, 2, 3]
    oversampled_residual_deshifting_list = [False]  # [True, False]
    oversampling_list = [1, 2, 3, 4, 5]
    step_factor_list = [0.2]  # [0.2, 0.5, 1]

    for step_factor in step_factor_list:
        for oversampled_residual_deshifting in oversampled_residual_deshifting_list:
            for oversampling in oversampling_list:
                for deshift_order in deshift_order_list:
                    kwargs_one_step = {'verbose': False,
                                       'oversampled_residual_deshifting': oversampled_residual_deshifting,
                                       'step_factor': step_factor,
                                       'deshift_order': deshift_order}
                    print(kwargs_one_step)

                    stack_psf_guassian_high_res(oversampling, num_stars, kwargs_one_step, num_iterations, n_recenter)