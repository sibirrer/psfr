from psfr.mask_util import mask_configuration
import numpy.testing as npt
import numpy as np


def test_mask_configuration():

    star1 = np.ones((11, 11))
    star2 = np.ones((11, 11)) * 10
    star_list = [star1, star2]

    # testing no mask
    mask_list = mask_configuration(star_list, mask_list=None, saturation_limit=None)
    npt.assert_almost_equal(star1, mask_list[0], decimal=6)
    npt.assert_almost_equal(star1, mask_list[1], decimal=6)

    # testing single saturation
    mask_list = mask_configuration(star_list, mask_list=None, saturation_limit=2)
    npt.assert_almost_equal(star1, mask_list[0], decimal=6)
    npt.assert_almost_equal(np.zeros_like(star2), mask_list[1], decimal=6)

    # testing list of saturation limits
    mask_list = mask_configuration(star_list, mask_list=None, saturation_limit=[0.1, 100])
    npt.assert_almost_equal(np.zeros_like(star1), mask_list[0], decimal=6)
    npt.assert_almost_equal(star1, mask_list[1], decimal=6)

    # testing raise with unequal saturation limits
    npt.assert_raises(ValueError, mask_configuration, star_list=star_list, mask_list=None, saturation_limit=[1, 1, 1])
