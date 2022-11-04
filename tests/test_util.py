import numpy.testing as npt
import numpy as np

from psfr import util
from lenstronomy.Util import util as lenstronomy_util


def test_regular2oversampled():
    from lenstronomy.LightModel.light_model import LightModel
    numpix = 41
    x_grid, y_grid = lenstronomy_util.make_grid(numPix=numpix, deltapix=1)
    gauss = LightModel(['GAUSSIAN'])
    kwargs_model = [{'amp': 1, 'sigma': 3, 'center_x': 0, 'center_y': 0}]
    flux_true = gauss.surface_brightness(x_grid, y_grid, kwargs_model)
    image = lenstronomy_util.array2image(flux_true)

    oversampling_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for oversampling in oversampling_list:
        image_oversampled = util.regular2oversampled(image, oversampling=oversampling)
        # check that surface brightness is conserved
        npt.assert_almost_equal(np.sum(image_oversampled), np.sum(image), decimal=5)
        n_pix = numpix * oversampling
        if n_pix % 2 == 0:
            n_pix -= 1
        # check length
        assert np.shape(image_oversampled) == (n_pix, n_pix)


def test_oversampled2data():
    x_grid, y_gird = lenstronomy_util.make_grid(19 * 5, 1., 1)
    sigma = 1.5
    amp = 2
    from lenstronomy.LightModel.Profiles.gaussian import Gaussian
    gaussian = Gaussian()
    flux = gaussian.function(x_grid, y_gird, amp=2, sigma=sigma)
    image_oversampled = lenstronomy_util.array2image(flux) / np.sum(flux) * amp

    for degrading_factor in range(7):
        oversampling = degrading_factor + 1
        kernel_degraded = util.oversampled2regular(image_oversampled, oversampling=oversampling)
        # kernel_degraded = kernel_util.degrade_kernel(kernel_super, degrading_factor=degrading_factor + 1)
        print(oversampling)
        npt.assert_almost_equal(np.sum(kernel_degraded), amp, decimal=8)


def test_regular2oversampled_inverse():
    from lenstronomy.LightModel.light_model import LightModel
    numpix = 41
    x_grid, y_grid = lenstronomy_util.make_grid(numPix=numpix, deltapix=1)
    gauss = LightModel(['GAUSSIAN'])
    kwargs_model = [{'amp': 1, 'sigma': 3, 'center_x': 0, 'center_y': 0}]
    flux_true = gauss.surface_brightness(x_grid, y_grid, kwargs_model)
    image = lenstronomy_util.array2image(flux_true)

    oversampling_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for oversampling in oversampling_list:
        image_oversampled = util.regular2oversampled(image, oversampling=oversampling)
        # degrade and compare with image
        image_degraded = util.oversampled2regular(image_oversampled, oversampling)
        assert np.shape(image_degraded) == np.shape(image)

        if False:
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            f, axes = plt.subplots(1, 3, figsize=(4 * 3, 4))
            vmin, vmax = -5, -1
            ax = axes[0]
            im = ax.imshow(np.log10(image_degraded), origin='lower', vmin=vmin, vmax=vmax)
            ax.autoscale(False)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title('image_degraded %f' % oversampling)

            ax = axes[1]
            im = ax.imshow(np.log10(image), origin='lower', vmin=vmin, vmax=vmax)
            ax.autoscale(False)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title('image')

            ax = axes[2]
            im = ax.imshow(image_degraded - image, origin='lower')
            ax.autoscale(False)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title('image')
            plt.show()
        if True:
            if oversampling % 2 == 1:
                npt.assert_almost_equal(image_degraded - image, 0, decimal=8)
            else:
                npt.assert_almost_equal(image_degraded - image, 0, decimal=3)


def test_jwst_example_stars():
    star_list_jwst = util.jwst_example_stars()
    assert len(star_list_jwst) == 5


def test_median_with_mask():
    mask_list = []
    data_list = []
    for i in range(5):
        mask = np.random.randint(2, size=(3, 3))
        data = np.ones((3, 3))
        data[mask == 0] = 2
        mask_list.append(mask)
        data_list.append(data)
    median_stack = util.median_with_mask(np.array(data_list), np.array(mask_list))
    npt.assert_almost_equal(median_stack, np.ones((3, 3)))

