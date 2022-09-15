from scipy import ndimage
import os
import astropy.io.fits as pyfits
from lenstronomy.Util import kernel_util, image_util
import psfr


def regular2oversampled(image, oversampling=1):
    """
    makes each pixel n x n pixels (with n=oversampling), makes it such that center remains in center pixel
    No sharpening below the original pixel scale is performed. This function should behave as the inverse of
    oversampled2regular(). This function is flux conserving.

    Parameters
    ----------
    image : 2d numpy array, square size with odd length n
        Data or model in regular pixel units of the data.
    oversampling : integer >= 1
        oversampling factor per axis of the output relative to the input image

    Returns
    -------
    image_oversampled : 2d numpy array, square size with odd length ns, with ns = oversampling * n for n odd,
        and ns = oversampling * n - 1 for n even
    """
    if oversampling == 1:
        return image
    image_oversampled = image.repeat(oversampling, axis=0).repeat(oversampling, axis=1)
    if oversampling % 2 == 1:
        pass  # this is already centered with odd total number of pixels
    else:
        # for even number super-sampling half a super-sampled pixel offset needs to be performed
        # we do the shift and cut in random directions such that it averages out

        image_oversampled1 = ndimage.shift(image_oversampled, shift=[-0.5, -0.5], order=1)
        # and the last column and row need to be removed
        image_oversampled1 = image_oversampled1[:-1, :-1]

        image_oversampled2 = ndimage.shift(image_oversampled, shift=[+0.5, +0.5], order=1)
        # and the last column and row need to be removed
        image_oversampled2 = image_oversampled2[1:, 1:]
        image_oversampled = (image_oversampled1 + image_oversampled2) / 2
    return image_oversampled / oversampling ** 2


def oversampled2regular(image_oversampled, oversampling=1):
    """
    Averages the pixel flux such that s x s oversampled pixels result in one pixel, with s = oversampling.
    The routine is designed to keep the centroid in the very centered pixel.

    This function should behave as the inverse of regular2oversampled().

    Parameters
    ----------
    image_oversampled : 2d numpy array, square size with odd length NxN
        Oversampled model
    oversampling : integer >= 1
        oversampling factor per axis of the input relative to the output

    Returns
    -------
    image_degraded : 2d numpy array, square size with odd length n
        with odd oversampling, n = N / oversampling, else n = (N + 1) / oversampling
        TODO: documentation here not accurate
    """
    image_degraded = kernel_util.degrade_kernel(image_oversampled, oversampling)
    # if oversampling == 1:
    #    return image_oversampled
    # if oversampling % 2 == 0:
    #    image_degraded = kernel_util.averaging_even_kernel(image_oversampled, oversampling)
    # else:
    #    image_degraded = kernel_util.averaging_odd_kernel(image_oversampled, oversampling)
    #    # degrading_factor**2  # multiplicative factor added when providing flux conservation
    #    image_degraded *= oversampling ** 2

    n = len(image_oversampled)
    # TODO: this should be in a single function and not compensating for kernel_util.degrade_kernel()
    if n % oversampling == 0:
        n_pix = int(n / oversampling)
        image_degraded = image_util.cut_edges(image_degraded, n_pix)
    return image_degraded


def jwst_example_stars():
    """
    imports example stars cutout from early JWST data

    Returns
    -------
    star_list : list of 2d arrays
        cutout stars from JWST data
    """
    package_path = psfr.__path__[0]
    path_stars = os.path.join(package_path, 'Data/JWST/')
    star_name = 'psf_f090w_star'

    star_list = []
    for i in range(5):
        path = os.path.join(path_stars, star_name+str(i)+'.fits')
        hdulist_star = pyfits.open(path)

        star = hdulist_star[0].data
        star_list.append(star)
    return star_list

