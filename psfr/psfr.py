"""Main module."""
import numpy as np
import scipy.optimize
from scipy.ndimage import interpolation
import matplotlib.pylab as plt
from lenstronomy.Util import util, kernel_util, image_util

# TODO: change this dependency
from lenstronomy.Workflow.psf_fitting import PsfFitting
combine_psf = PsfFitting.combine_psf


def stack_psf(star_list, oversampling=1, saturation_limit=None, num_iteration=5, n_recenter=10,
              verbose=False, mask_list=None, kwargs_one_step={}, **kwargs):
    """
    Parameters
    ----------
    star_list: list of numpy arrays (2D) of stars.
     Odd square axes with the approximate star within the center pixel. All stars need the same cutout size.
    oversampling: integer, higher-resolution PSF reconstruction and return
    saturation_limit: float or list of floats for each star;
     pixel values abouve this threshold will not be considered in the reconstruction.
    n_recenter: integer, every n_recenter iterations of the updated PSF, a re-centering of the centroids are performed with the updated PSF guess
    """

    n_star = len(star_list)
    # define saturation masks
    if saturation_limit is not None:
        n_sat = len(np.atleast_1d(saturation_limit))
        if n_sat == 1:
            saturation_limit = [saturation_limit] * n_star
        else:
            if n_sat != n_star:
                raise ValueError('saturation_limit is a list with nonequal length to star_list.')
    # initiate a mask_list that accepts all pixels
    if mask_list is None:
        mask_list = []
        for i, star in enumerate(star_list):
            mask = np.ones_like(star)
            mask_list.append(mask)
    # add threshold for saturation in mask
    if saturation_limit is not None:
        for i, mask in enumerate(mask_list):
            mask[star_list[i] > saturation_limit[i]] = 0

            # define base stacking without shift
    # integer shift
    # stacking with mask weight
    star_stack_base = np.zeros_like(star_list[0])
    weight_map = np.zeros_like(star_list[0])
    for i, star in enumerate(star_list):
        star_stack_base += star * mask_list[i]
        weight_map += mask_list[i] * np.sum(star)
    star_stack_base = star_stack_base / weight_map
    star_stack_base /= np.sum(star_stack_base)

    # estimate center offsets based on base stacking
    center_list = []
    for i, star in enumerate(star_list):
        x_c, y_c = _fit_centroid(star, star_stack_base, mask_list[i])
        center_list.append([x_c, y_c])
    if verbose:
        print(center_list, 'center_list')

    # simultaneous iterative correction of PSF starting with base stacking in oversampled resolution
    psf_guess = star_stack_base.repeat(oversampling, axis=0).repeat(oversampling, axis=1)
    if oversampling % 2 == 1:
        pass
    else:
        # for even number super-sampling half a super-sampled pixel offset needs to be performed
        psf_guess1 = interpolation.shift(psf_guess, [- 0.5, - 0.5], order=1)
        # and the last column and row need to be removed
        psf_guess1 = psf_guess1[:-1, :-1]

        psf_guess2 = interpolation.shift(psf_guess, [+0.5, +0.5], order=1)
        # and the last column and row need to be removed
        psf_guess2 = psf_guess2[1:, 1:]
        psf_guess = (psf_guess1 + psf_guess2) / 2

    # psf_guess = kernel_util.subgrid_kernel(star_stack_base, oversampling, odd=True, num_iter=0)
    if verbose:
        plt.imshow(np.log10(star_stack_base), origin='lower')
        plt.title('star_stack_base')
        plt.show()

        plt.imshow(np.log10(psf_guess), origin='lower')
        plt.title('input first guess')
        plt.show()

    for j in range(num_iteration):
        psf_guess = one_step_psf_estimate(star_list, psf_guess, center_list, mask_list,
                                          oversampling=oversampling, **kwargs_one_step)
        if j % n_recenter == 0 and j != 0:
            center_list = []
            for i, star in enumerate(star_list):
                x_c, y_c = _fit_centroid(star, psf_guess, mask_list[i], oversampling=oversampling)
                center_list.append([x_c, y_c])
        if verbose:
            plt.imshow(np.log(psf_guess), vmin=-5, vmax=-1)
            plt.title('iteration %s' % j)
            plt.show()
    return psf_guess, center_list, mask_list


def one_step_psf_estimate(star_list, psf_guess, center_list, mask_list, error_map_list=None, step_factor=0.2,
                          oversampling=1, verbose=False):
    """

    """
    if mask_list is None:
        mask_list = []
        for i, star in enumerate(star_list):
            mask_list.append(np.ones_like(star))
    if error_map_list is None:
        error_map_list = [None] * len(star_list)
    psf_list_new = []
    for i, star in enumerate(star_list):
        center = center_list[i]
        # shift PSF guess to estimated position of star

        # x_int = int(round(center[0]))
        # y_int = int(round(center[1]))
        # x_int, y_int = 0, 0

        if verbose:
            plt.imshow(np.log10(star), origin='lower', vmin=-5, vmax=-1)
            plt.title('star 1')
            plt.show()
        if verbose:
            plt.imshow(np.log10(psf_guess), origin='lower', vmin=-5, vmax=-1)
            plt.title('psf guess')
            plt.show()

        # shift PSF to position pre-determined to be the center of the point source, and degrate it to the image
        if oversampling > 1:
            psf_shifted = shift_psf(psf_guess, oversampling, shift=center_list[i], degrade=False, n_pix_star=len(star))
            if verbose:
                plt.imshow(np.log10(psf_shifted), origin='lower', vmin=-5, vmax=-1)
                plt.title('psf shifted')
                plt.show()
            # make data degraded version
            psf_shifted_data = kernel_util.degrade_kernel(psf_shifted, degrading_factor=oversampling)
            # make sure size is the same as the data
            psf_shifted_data = kernel_util.cut_psf(psf_shifted_data, len(star))
        else:
            psf_shifted_data = shift_psf(psf_guess, oversampling, shift=center_list[i], degrade=True,
                                         n_pix_star=len(star))
            psf_shifted = psf_shifted_data

        if verbose:
            plt.imshow(np.log10(psf_shifted_data), origin='lower', vmin=-5, vmax=-1)
            plt.title('psf shifted data')
            plt.show()
        # linear inversion in 1d

        amp = _linear_amplitude(star, psf_shifted_data, variance=error_map_list[i], mask=mask_list[i])

        # TODO make residual map on higher resolution space
        # compute residuals on data
        if False:  # directly in oversampled space
            star_super = _image2oversampled(star, oversampling=oversampling)  # TODO: needs only be calculated once!
            mask_super = _image2oversampled(mask_list[i], oversampling=oversampling)
            mask_super[mask_super < 1] = 0
            mask_super[mask_super >= 1] = 1
            residuals_oversampled = (star_super - amp * psf_shifted) * mask_super

            # shift residuals back on higher res grid
            # inverse shift residuals
            shift_x = center[0] * oversampling
            shift_y = center[1] * oversampling

            residuals_shifted = interpolation.shift(residuals_oversampled, [-shift_y, -shift_x], order=0)

        else:  # in data space and then being oversampled
            residuals = (star - amp * psf_shifted_data) * mask_list[i]

            if verbose:
                plt.imshow(star - amp * psf_shifted_data, origin='lower')
                plt.title('residuals')
                plt.show()

            if verbose:
                plt.imshow(residuals, origin='lower')
                plt.title('residuals')
                plt.show()
            # renormalize residuals
            residuals /= amp  # devide by amplitude of point source
            # high-res version of residuals
            residuals_oversampled = residuals.repeat(oversampling, axis=0).repeat(oversampling, axis=1)
            if verbose:
                plt.imshow(residuals_oversampled, origin='lower')
                plt.title('residuals oversampled')
                plt.show()
            # shift residuals back on higher res grid
            # inverse shift residuals
            shift_x = center[0] * oversampling
            shift_y = center[1] * oversampling
            # for odd number super-sampling
            if oversampling % 2 == 1:
                residuals_shifted = interpolation.shift(residuals_oversampled, [-shift_y, -shift_x], order=0)

            else:
                # for even number super-sampling half a super-sampled pixel offset needs to be performed
                # TODO: move them in random direction in all four directions (not only two)
                rand_num = np.random.randint(2)
                if rand_num == 1:
                    residuals_shifted = interpolation.shift(residuals_oversampled, [-shift_y - 0.5, -shift_x - 0.5],
                                                            order=0)
                    # and the last column and row need to be removed
                    residuals_shifted = residuals_shifted[:-1, :-1]
                else:
                    residuals_shifted = interpolation.shift(residuals_oversampled, [-shift_y + 0.5, -shift_x + 0.5],
                                                            order=0)
                    # and the last column and row need to be removed
                    residuals_shifted = residuals_shifted[1:, 1:]

        # re-size shift residuals
        psf_size = len(psf_guess)
        residuals_shifted = image_util.cut_edges(residuals_shifted, psf_size)
        if verbose:
            plt.imshow(residuals_shifted, origin='lower')
            plt.title('residuals shifted')
            plt.show()

        # normalize residuals
        correction = residuals_shifted - np.mean(residuals_shifted)
        if verbose:
            plt.imshow(correction, origin='lower')
            plt.title('correction')
            plt.show()
        psf_new = psf_guess + correction
        psf_new[psf_new < 0] = 0
        psf_new /= np.sum(psf_new)
        if verbose:
            plt.imshow(psf_new, origin='lower')
            plt.title('psf_new')
            plt.show()
        psf_list_new.append(psf_new)

    # stack all residuals and update the psf guess
    # TODO: make combine_psf remember the masks and relative brightness in the weighting scheme (for later)
    kernel_new = combine_psf(psf_list_new, psf_guess, factor=step_factor, stacking_option='mean', symmetry=1)
    kernel_new = kernel_util.cut_psf(kernel_new, psf_size=len(psf_guess))
    return kernel_new


def shift_psf(psf_center, oversampling, shift, degrade=True, n_pix_star=None):
    """
    shift PSF to the star position. Optionally degrades to the image resolution afterwards

    Parameters
    ----------
    psf_center : 2d numpy array with odd square length
        Centered PSF in the oversampling space of the input
    oversampling : integer >= 1
        oversampling factor per axis of the psf_center relative to the data and coordinate shift
    shift : [x, y], 2d floats
        off-center shift in the units of the data
    degrade : boolean
        if True degrades the shifted PSF to the data resolution and cuts the resulting size to n_pix_star
    n_pix_star : odd integer
        size per axis of the data, used when degrading the shifted {SF

    Returns
    -------
    psf_shifted : 2d numpy array, odd axis number
        shifted PSF, optionally degraded to the data resolution

    """
    shift_x = shift[0] * oversampling
    shift_y = shift[1] * oversampling
    # shift psf
    # TODO: what is optimal interpolation in the shift, or doing it in Fourier space instead?
    psf_shifted1 = interpolation.shift(psf_center, [shift_y, shift_x], order=1)
    psf_shifted2 = interpolation.shift(psf_center, [shift_y, shift_x], order=0)
    psf_shifted = (psf_shifted1 + psf_shifted2) / 2

    # resize to pixel scale (making sure the grid definition with the center in the central pixel is preserved)
    if degrade is True:
        psf_shifted = kernel_util.degrade_kernel(psf_shifted, degrading_factor=oversampling)
        psf_shifted = kernel_util.cut_psf(psf_shifted, n_pix_star)
    return psf_shifted


def _linear_amplitude(data, model, variance=None, mask=None):
    """
    computes linear least square amplitude to minimize
    min[(data - amp * model)^2 / variance]

    Parameters
    ----------
    data : 2d numpy array
        the measured data (i.e. of a star or other point-like source)
    model: 2d numpy array, same size as data
        model prediction of the data, modulo a linear amplitude scaling
        (i.e. a PSF model that is sub-pixel shifted to match the astrometric position of the star in the data)
    variance : None, or 2d numpy array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels
    mask : None, or 2d integer or boolean array, same size as data
        Masking pixels not to be considered in fitting the linear amplitude;
        zeros are ignored, ones are taken into account. None as input takes all pixels into account (default)

    Returns
    -------
    amp : float
        linear amplitude such that (data - amp * model)^2 / variance is minimized
    """
    y = util.image2array(data)
    x = util.image2array(model)
    if mask is not None:
        mask_ = util.image2array(mask)
        x = x[mask_ == 1]
        y = y[mask_ == 1]

    if variance is None:
        w = 1  # we simply give equal weight to all data points
    else:
        w = util.image2array(1. / variance)
        if mask is not None:
            w = w[mask == 1]
    wsum = np.sum(w)
    xw = np.sum(w * x) / wsum
    yw = np.sum(w * y) / wsum
    amp = np.sum(w * (x - xw) * (y - yw)) / np.sum(w * (x - xw) ** 2)
    return amp


def _fit_centroid(data, model, mask=None, variance=None, oversampling=1):
    """
    fit the centroid of the model to the image by shifting and scaling the model to best match the data
    This is done in a non-linear minimizer in the positions (x, y) and linear amplitude minimization on the fly.
    The model is interpolated to match the data. The starting point of the model is to be aligned with the image.

    Parameters
    ----------
    data : 2d numpy array
        data of a point-like source for which a centroid position is required
    model : 2d numpy array, odd squared length
        a centered model of the PSF in oversampled space
    mask : None, or 2d integer or boolean array, same size as data
        Masking pixels not to be considered in fitting the linear amplitude;
        zeros are ignored, ones are taken into account. None as input takes all pixels into account (default)
    variance : None, or 2d numpy array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels
    oversampling : integer >= 1
        oversampling factor per axis of the model relative to the data and coordinate shift

    Returns
    -------
    center_shift : 2d array (delta_x, delta_y)
        required shift in units of pixels in the data such that the model matches best the data
    """

    def _minimize(x, data, model, variance=None, mask=None, oversampling=1):
        # shift model to proposed astrometric position
        model_shifted = shift_psf(psf_center=model, oversampling=oversampling, shift=x, degrade=True,
                                  n_pix_star=len(data))
        # linear amplitude
        if mask is None:
            mask = np.ones_like(data)
        amp = _linear_amplitude(data, model_shifted, variance=variance, mask=mask)

        if variance is None:
            variance = 1
        # compute chi2
        chi2 = np.sum((data - model_shifted * amp) ** 2 / variance * mask)
        return chi2

    init = np.array([0, 0])
    x = scipy.optimize.minimize(_minimize, init, args=(data, model, variance, mask, oversampling),
                                bounds=((-2, 2), (-2, 2)), method='Nelder-Mead')
    return x.x


def _image2oversampled(image, oversampling=1):
    """
    makes each pixel n x n pixels (with n=oversampling), makes it such that center remains in center pixel
    """
    if oversampling == 1:
        return image
    image_oversampled = image.repeat(oversampling, axis=0).repeat(oversampling, axis=1)
    if oversampling % 2 == 1:
        pass  # this is already centered with odd total number of pixels

    else:
        # for even number super-sampling half a super-sampled pixel offset needs to be performed
        # we do the shift and cut in random directions such that it averages out
        rand_num = np.random.randint(2)
        if rand_num == 0:
            image_oversampled = interpolation.shift(image_oversampled, [-0.5, -0.5], order=0)
            # and the last column and row need to be removed
            image_oversampled = image_oversampled[:-1, :-1]
        else:
            image_oversampled = interpolation.shift(image_oversampled, [+0.5, +0.5], order=0)
            # and the last column and row need to be removed
            image_oversampled = image_oversampled[1:, 1:]
    return image_oversampled

