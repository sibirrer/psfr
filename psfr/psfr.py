"""Main module."""
import numpy as np
import scipy.optimize
from scipy import ndimage
import matplotlib.pylab as plt
import matplotlib.animation as animation
from lenstronomy.Util import util, kernel_util, image_util
from psfr.util import regular2oversampled, oversampled2regular
from psfr import mask_util
from psfr import verbose_util

from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer


def stack_psf(star_list, oversampling=1, mask_list=None, error_map_list=None, saturation_limit=None, num_iteration=5,
              n_recenter=10,
              verbose=False, kwargs_one_step=None, psf_initial_guess=None, kwargs_psf_stacking=None,
              centroid_optimizer='Nelder-Mead', **kwargs_animate):
    """
    Parameters
    ----------
    star_list: list of numpy arrays (2D) of stars. Odd square axis shape.
        Cutout stars from images with approximately centered on the center pixel. All stars need the same cutout size.
    oversampling : integer >=1
        higher-resolution PSF reconstruction and return
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included
    error_map_list : None, or list of 2d numpy array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels.
        If not set, assumes equal variances for all pixels.
    saturation_limit: float or list of floats of length of star_list
        pixel values above this threshold will not be considered in the reconstruction.
    num_iteration : integer >= 1
        number of iterative corrections applied on the PSF based on previous guess
    n_recenter: integer
        Every n_recenter iterations of the updated PSF, a re-centering of the centroids are performed with the updated
        PSF guess.
    verbose : boolean
        If True, provides plots of updated PSF during the iterative process
    kwargs_one_step : keyword arguments to be passed to one_step_psf_estimate() method
        See one_step_psf_estimate() method for options
    psf_initial_guess : None or 2d numpy array with square odd axis
        Initial guess PSF on oversampled scale. If not provided, estimates an initial guess with the stacked stars.
    kwargs_animate : keyword arguments for animation settings
        Settings to display animation of interative process of psf reconstuction. The argument is organized as:
            {animate: bool, output_dir: str (directory to save animation in),
            duration: int (length of animation in milliseconds)}
    kwargs_psf_stacking: keyword argument list of arguments going into combine_psf()
        stacking_option : option of stacking, 'mean' or 'median'
        symmetry: integer, imposed symmetry of PSF estimate
    centroid_optimizer: 'Nelder-Mead' or 'PSO'
        option for the optimizing algorithm used to find the center of each PSF in data.


    Returns
    -------
    psf_guess : 2d numpy array with square odd axis
        best guess PSF in the oversampled resolution
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included.
        This list is updated with all the criteria applied on the fitting and might deviate from the input mask_list.
    center_list : list of 2d floats
        list of astrometric centers relative to the center pixel of the individual stars

    """
    # update the mask according to settings
    mask_list = mask_util.mask_configuration(star_list, mask_list=mask_list, saturation_limit=saturation_limit)
    if kwargs_one_step is None:
        kwargs_one_step = {}
    if kwargs_psf_stacking is None:
        kwargs_psf_stacking = {}
    if error_map_list is None:
        error_map_list = [None] * len(star_list)

    # update default options for animations
    animation_options = {'animate': False, 'output_dir': 'stacked_psf_animation.gif', 'duration': 5000}
    animation_options.update(kwargs_animate)
    # define base stacking without shift offset shifts
    # stacking with mask weight
    star_stack_base = base_stacking(star_list, mask_list)
    star_stack_base /= np.sum(star_stack_base)

    # estimate center offsets based on base stacking PSF estimate
    center_list = []
    for i, star in enumerate(star_list):
        x_c, y_c = centroid_fit(star, star_stack_base, mask_list[i], optimizer_type=centroid_optimizer,
                                variance=error_map_list[i])
        center_list.append([x_c, y_c])

    if psf_initial_guess is None:
        psf_guess = regular2oversampled(star_stack_base, oversampling=oversampling)
    else:
        psf_guess = psf_initial_guess

    if verbose:
        f, axes = plt.subplots(1, 1, figsize=(4 * 2, 4))
        ax = axes
        ax.imshow(np.log10(psf_guess), origin='lower')
        ax.set_title('input first guess')
        plt.show()

    # simultaneous iterative correction of PSF starting with base stacking in oversampled resolution
    images_to_animate = []

    for j in range(num_iteration):
        psf_guess = one_step_psf_estimate(star_list, psf_guess, center_list, mask_list, error_map_list=error_map_list,
                                          oversampling=oversampling, **kwargs_psf_stacking, **kwargs_one_step)
        if j % n_recenter == 0 and j != 0:
            center_list = []
            for i, star in enumerate(star_list):
                x_c, y_c = centroid_fit(star, psf_guess, mask_list[i], oversampling=oversampling,
                                        variance=error_map_list[i], optimizer_type=centroid_optimizer)
                center_list.append([x_c, y_c])
        if animation_options['animate']:
            images_to_animate.append(psf_guess)
        if verbose:
            plt.imshow(np.log(psf_guess), vmin=-5, vmax=-1)
            plt.title('iteration %s' % j)
            plt.colorbar()
            plt.show()

    # function that is called to update the image for the animation
    def _updatefig(i):
        img.set_data(np.log10(images_to_animate[i]))
        return [img]

    if animation_options['animate']:
        global anim
        fig = plt.figure()
        img = plt.imshow(np.log10(images_to_animate[0]))
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='k', alpha=1.)
        cmap.set_under('k')
        # animate and display iterative psf reconstuction
        anim = animation.FuncAnimation(fig, _updatefig, frames=len(images_to_animate),
                                       interval=int(animation_options['duration'] / len(images_to_animate)), blit=True)
        anim.save(animation_options['output_dir'])
        plt.close()

    return psf_guess, center_list, mask_list


def one_step_psf_estimate(star_list, psf_guess, center_list, mask_list, error_map_list=None, oversampling=1,
                          step_factor=0.2, oversampled_residual_deshifting=False, deshift_order=1, verbose=False,
                          **kwargs_psf_stacking):
    """

    Parameters
    ----------
    star_list: list of numpy arrays (2D) of stars. Odd square axis shape.
        Cutout stars from images with approximately centered on the center pixel. All stars need the same cutout size.
    psf_guess : 2d numpy array with square odd axis
        best guess PSF in the oversampled resolution prior to this iteration step
    center_list : list of 2d floats
        list of astrometric centers relative to the center pixel of the individual stars
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included
    oversampling : integer >=1
        higher-resolution PSF reconstruction and return
    error_map_list : None, or list of 2d numpy array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels.
        If not set, assumes equal variances for all pixels.
    oversampled_residual_deshifting : boolean
        if True; produces first an oversampled residual map and then de-shifts it back to the center for each star
        if False; produces a residual map in the data space and de-shifts it into a higher resolution residual map for
        stacking
    deshift_order : integer >= 0
        polynomial order of interpolation of the de-shifting of the residuals to the center to be interpreted as
        desired corrections for a given star
    step_factor : float or integer in (0, 1]
        weight of updated estimate based on new and old estimate;
        psf_update = step_factor * psf_new + (1 - step_factor) * psf_old
    kwargs_psf_stacking: keyword argument list of arguments going into combine_psf()
        stacking_option : option of stacking, 'mean' or 'median'
        symmetry: integer, imposed symmetry of PSF estimate
    verbose : boolean
        If True, provides plots of intermediate products walking through one iteration process for each individual star
    """
    if mask_list is None:
        mask_list = []
        for i, star in enumerate(star_list):
            mask_list.append(np.ones_like(star))
    if error_map_list is None:
        error_map_list = [None] * len(star_list)
    psf_list_new = []
    weight_list = []
    for i, star in enumerate(star_list):
        center = center_list[i]
        # shift PSF guess to estimated position of star
        # shift PSF to position pre-determined to be the center of the point source, and degrate it to the image
        psf_shifted = shift_psf(psf_guess, oversampling, shift=center_list[i], degrade=False, n_pix_star=len(star))

        # make data degraded version
        psf_shifted_data = oversampled2regular(psf_shifted, oversampling=oversampling)
        # make sure size is the same as the data and normalized to sum = 1
        psf_shifted_data = kernel_util.cut_psf(psf_shifted_data, len(star))
        # linear inversion in 1d
        amp = _linear_amplitude(star, psf_shifted_data, variance=error_map_list[i], mask=mask_list[i])
        weight_list.append(amp)

        # compute residuals on data
        if oversampled_residual_deshifting:  # directly in oversampled space
            star_super = regular2oversampled(star, oversampling=oversampling)  # TODO: needs only be calculated once!
            mask_super = regular2oversampled(mask_list[i], oversampling=oversampling)
            # attention the routine is flux conserving and need to be changed for the mask,
            # in case of interpolation we block everything that has a tenth of a mask in there
            mask_super[mask_super < 1. / oversampling ** 2 / 10] = 0
            mask_super[mask_super >= 1. / oversampling ** 2 / 10] = 1
            residuals = (star_super - amp * psf_shifted) * mask_super
            residuals /= amp

            # shift residuals back on higher res grid
            # inverse shift residuals
            shift_x = center[0] * oversampling
            shift_y = center[1] * oversampling
            residuals_shifted = ndimage.shift(residuals, shift=[-shift_y, -shift_x], order=deshift_order)

        else:  # in data space and then being oversampled
            residuals = (star - amp * psf_shifted_data) * mask_list[i]
            # re-normalize residuals
            residuals /= amp  # divide by amplitude of point source
            # high-res version of residuals
            residuals = residuals.repeat(oversampling, axis=0).repeat(oversampling, axis=1) / oversampling ** 2
            # shift residuals back on higher res grid
            # inverse shift residuals
            shift_x = center[0] * oversampling
            shift_y = center[1] * oversampling

            if oversampling % 2 == 1:  # for odd number super-sampling
                residuals_shifted = ndimage.shift(residuals, shift=[-shift_y, -shift_x], order=deshift_order)

            else:  # for even number super-sampling
                # for even number super-sampling half a super-sampled pixel offset needs to be performed
                # TODO: move them in all four directions (not only two)
                residuals_shifted1 = ndimage.shift(residuals, shift=[-shift_y - 0.5, -shift_x - 0.5],
                                                   order=deshift_order)
                # and the last column and row need to be removed
                residuals_shifted1 = residuals_shifted1[:-1, :-1]

                residuals_shifted2 = ndimage.shift(residuals, shift=[-shift_y + 0.5, -shift_x + 0.5],
                                                   order=deshift_order)
                # and the last column and row need to be removed
                residuals_shifted2 = residuals_shifted2[1:, 1:]
                residuals_shifted = (residuals_shifted1 + residuals_shifted2) / 2

        # re-size shift residuals
        psf_size = len(psf_guess)
        residuals_shifted = image_util.cut_edges(residuals_shifted, psf_size)
        # normalize residuals
        correction = residuals_shifted - np.mean(residuals_shifted)
        psf_new = psf_guess + correction
        psf_new[psf_new < 0] = 0
        psf_new /= np.sum(psf_new)
        if verbose:
            fig = verbose_util.verbose_one_step(star, psf_shifted, psf_shifted_data, residuals, residuals_shifted,
                                                correction, psf_new)
            fig.show()
        psf_list_new.append(psf_new)

    # stack all residuals and update the psf guess
    # TODO: make combine_psf remember the masks and relative brightness in the weighting scheme (for later)
    psf_stacking_options = {'stacking_option': 'median', 'symmetry': 1}
    psf_stacking_options.update(kwargs_psf_stacking)
    weight_list = np.array(weight_list)
    weight_list[weight_list < 0] = 0
    # the weights are supposed to be Poisson noise dominated, which scales as the square root of the brightness
    weights_psf_combined = np.sqrt(weight_list)
    kernel_new = combine_psf(psf_list_new, psf_guess, factor=step_factor, mask_list=mask_list,
                             weight_list=weights_psf_combined, **psf_stacking_options)
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
    # Partial answer: interpolation in order=1 is better than order=0 (order=2 is even better for Gaussian PSF's)
    psf_shifted = ndimage.shift(psf_center, shift=[shift_y, shift_x], order=2)

    # resize to pixel scale (making sure the grid definition with the center in the central pixel is preserved)
    if degrade is True:
        psf_shifted = oversampled2regular(psf_shifted, oversampling=oversampling)
        psf_shifted = image_util.cut_edges(psf_shifted, n_pix_star)
        # psf_shifted = kernel_util.cut_psf(psf_shifted, n_pix_star)
    return psf_shifted


def base_stacking(star_list, mask_list):
    """
    Basic stacking of stars in luminosity-weighted and mask-excluded regime.
    The method ignores sub-pixel off-centering of individual stars nor does it provide an oversampled solution.
    This method is intended as a baseline comparison and as an initial guess version for the full PSF-r features.

    Parameters
    ----------
    star_list : list of 2d numpy arrays
        list of cutout stars (to be included in the fitting)
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included

    Returns
    -------
    star_stack_base : 2d numpy array of size of the stars in star_list
        luminosity weighted and mask-excluded stacked stars
    """
    star_stack_base = np.zeros_like(star_list[0])
    weight_map = np.zeros_like(star_list[0])

    for i, star in enumerate(star_list):
        star_stack_base += star * mask_list[i]
        weight_map += mask_list[i] * np.sum(star)

    ###code can't handle situations where there is never a non-zero pixel
    weight_map[weight_map == 0] = 1e-6

    star_stack_base = star_stack_base / weight_map

    return star_stack_base


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
    else:
        mask_ = None

    if variance is None:
        w = 1  # we simply give equal weight to all data points
    else:
        w = util.image2array(1. / variance)
        if mask_ is not None:  # mask is not None:
            w = w[mask_ == 1]
    wsum = np.sum(w)
    xw = np.sum(w * x) / wsum
    yw = np.sum(w * y) / wsum
    amp = np.sum(w * (x - xw) * (y - yw)) / np.sum(w * (x - xw) ** 2)
    return amp


def centroid_fit(data, model, mask=None, variance=None, oversampling=1, optimizer_type='Nelder-Mead'):
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
    optimizer_type: 'Nelder-Mead' or 'PSO'

    Returns
    -------
    center_shift : 2d array (delta_x, delta_y)
        required shift in units of pixels in the data such that the model matches best the data
    """

    def _minimize(x, data, model, mask, variance, oversampling=1, negative=1):
        # shift model to proposed astrometric position
        model_shifted = shift_psf(psf_center=model, oversampling=oversampling, shift=x, degrade=True,
                                  n_pix_star=len(data))
        # linear amplitude
        amp = _linear_amplitude(data, model_shifted, variance=variance, mask=mask)

        chi2 = negative * np.sum((data - model_shifted * amp) ** 2 / variance * mask)

        return chi2

    init = np.array([0, 0])
    bounds = ((-5, 5), (-5, 5))
    if mask is None:
        mask = np.ones_like(data)
    if variance is None:
        variance = np.ones_like(data)

    if optimizer_type == 'Nelder-Mead':
        x = scipy.optimize.minimize(_minimize, init, args=(data, model, mask, variance, oversampling),
                                    bounds=bounds, method='Nelder-Mead')
        return x.x

    elif optimizer_type == 'PSO':
        lowerLims = np.array([bounds[0][0], bounds[1][0]])
        upperLims = np.array([bounds[0][1], bounds[1][1]])

        n_particles = 50
        n_iterations = 100
        pool = None
        pso = ParticleSwarmOptimizer(_minimize,
                                     lowerLims, upperLims, n_particles,
                                     pool=pool, args=[data, model, variance, mask],
                                     kwargs={'oversampling': oversampling,
                                             'negative': -1})

        result, [log_likelihood_list, pos_list, vel_list] = pso.optimize(n_iterations, verbose=False)
        return result

    else:
        raise ValueError('optimization type %s is not supported. Please use Nelder-Mead or PSO' % optimizer_type)


def combine_psf(kernel_list_new, kernel_old, mask_list=None, weight_list=None, factor=1., stacking_option='median',
                symmetry=1, combine_with_old=False):
    """
    updates psf estimate based on old kernel and several new estimates

    Parameters
    ----------
    kernel_list_new : list of 2d numpy arrays
        new PSF kernels estimated from the point sources in the image (un-normalized)
    kernel_old : 2d numpy array of shape of the oversampled kernel
        old PSF kernel
    mask_list : None or list of booleans of shape of kernel_list_new
        masks used in the 'kernel_list_new' determination. These regions will not be considered in the combined PSF.
    weight_list : None or list of floats with positive semi-definite values
        weights of the different new kernel estimates (i.e. brightness of the stars, etc)
    factor : weight of updated estimate based on new and old estimate, factor=1 means new estimate,
        factor=0 means old estimate
    stacking_option : string
        option of stacking, mean or median
    symmetry: integer >= 1
        imposed symmetry of PSF estimate
    combine_with_old : boolean
        if True, adds the previous PSF as one of the proposals for the new PSFs

    Returns
    -------
    kernel_return : updated PSF estimate and error_map associated with it
    """

    n = int(len(kernel_list_new) * symmetry)

    if weight_list is None:
        weight_list = np.ones(len(kernel_list_new))

    angle = 360. / symmetry
    kernelsize = len(kernel_old)
    kernel_list = np.zeros((n, kernelsize, kernelsize))
    weights = np.zeros(n)
    i = 0
    for j, kernel_new in enumerate(kernel_list_new):

        for k in range(symmetry):
            kernel_rotated = image_util.rotateImage(kernel_new, angle * k)
            kernel_norm = kernel_util.kernel_norm(kernel_rotated)
            kernel_list[i, :, :] = kernel_norm
            weights[i] = weight_list[j]
            i += 1

    if combine_with_old is True:
        kernel_old_rotated = np.zeros((symmetry, kernelsize, kernelsize))
        for i in range(symmetry):
            kernel_old_rotated[i, :, :] = kernel_old / np.sum(kernel_old)
            kernel_list = np.append(kernel_list, kernel_old_rotated, axis=0)
            weights = np.append(weights, 1)

    # TODO: add mask_list
    # TODO: outlier detection?
    if stacking_option == 'median_weight':
        # TODO: this is rather slow as it needs to loop through all the pixels
        # adapted from this thread:
        # https://stackoverflow.com/questions/26102867/python-weighted-median-algorithm-with-pandas
        flattened_psfs = np.array([y.flatten() for y in kernel_list])
        x_dim, y_dim = kernel_list[0].shape
        new_img = []
        for i in range(x_dim * y_dim):
            pixels = flattened_psfs[:, i]
            cumsum = np.cumsum(weights)
            cutoff = np.sum(weights) / 2.0
            pixels = np.sort(pixels)
            median = pixels[cumsum >= cutoff][0]
            new_img.append(median)
        kernel_new = np.array(new_img).reshape(x_dim, y_dim)

    elif stacking_option == 'mean':
        kernel_new = np.average(kernel_list, weights=weights, axis=0)

    elif stacking_option == 'median':
        kernel_new = np.median(kernel_list, axis=0)
    else:
        raise ValueError(" stack_option must be 'median', 'median_weight' or 'mean', %s is not supported."
                         % stacking_option)
    kernel_new = np.nan_to_num(kernel_new)
    kernel_new[kernel_new < 0] = 0
    kernel_new = kernel_util.kernel_norm(kernel_new)
    kernel_return = factor * kernel_new + (1. - factor) * kernel_old
    return kernel_return
