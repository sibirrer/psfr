from psfr import psfr
import numpy.testing as npt
import os
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt

def test_retrieve_psf():
    module_path = os.path.dirname(psfr.__file__)
    psf_filename = module_path + '/Data/JWST_mock/psf_f090w_supersample5_crop.fits'
    npt.assert_equal(os.path.isfile(psf_filename),True, err_msg = 'Psf file not found')

def test_reconstruct_psf():
    import lenstronomy.Util.kernel_util as util
    module_path = os.path.dirname(psfr.__file__)
    psf_filename = module_path + '/Data/JWST_mock/psf_f090w_supersample5_crop.fits'
    kernel = pyfits.getdata(psf_filename)

    oversampling = 5
    star_list_webb = []
    for i in range(5):
        x_shift, y_shift = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        star = psfr.shift_psf(psf_center=kernel, oversampling=5, shift=[x_shift, y_shift], degrade=True, n_pix_star=kernel.shape[0]/oversampling)
        star_list_webb.append(star)

    psf_psfr_super, center_list_psfr_super, mask_list = psfr.stack_psf(star_list_webb, oversampling=oversampling, 
                                                  saturation_limit=None, num_iteration=10, 
                                                  n_recenter=20)
                                                
    kernel_degraded = util.degrade_kernel(kernel, oversampling)
    psf_guess = star_list_webb[0]
    stacked_psf_degraded = psfr.oversampled2regular(psf_psfr_super, oversampling)

    diff1 = np.sum((psf_guess - kernel_degraded)**2)
    diff2 = np.sum((stacked_psf_degraded - kernel_degraded)**2)
    npt.assert_array_less(diff2, diff1, err_msg='reconstructed psf is worse than guess')

