from psfr import psfr
import numpy.testing as npt
import os
import astropy.io.fits as pyfits

def test_retrieve_psf():
    module_path = os.path.dirname(psfr.__file__)
    psf_filename = module_path + '\Data\JWST_mock\psf_f090w_supersample5_crop.fits'
    # kernel = pyfits.get_data(psf_filename)
    npt.assert_equal(os.path.isfile(psf_filename),True, err_msg = 'Psf file not found')