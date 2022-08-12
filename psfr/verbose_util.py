import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def verbose_one_step(star, psf_shifted, psf_shifted_data, residuals, residuals_shifted, correction, psf_new):
    """
    plotting of intermediate products happening during one step of the PSF iteration
    """

    fig, axes = plt.subplots(1, 7, figsize=(4 * 7, 4))
    vmin, vmax = -5, -1

    ax = axes[0]
    im = ax.imshow(np.log10(star / np.sum(star)), origin='lower', vmin=vmin, vmax=vmax)
    # ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('star normalized')

    ax = axes[1]
    im = ax.imshow(np.log10(psf_shifted), origin='lower', vmin=vmin, vmax=vmax)
    # ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('psf shifted (oversampled)')

    ax = axes[2]
    im = ax.imshow(np.log10(psf_shifted_data), origin='lower', vmin=vmin, vmax=vmax)
    # ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('psf shifted (regular)')

    ax = axes[3]
    im = ax.imshow(residuals, origin='lower')
    # ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('residuals on data')

    ax = axes[4]
    im = ax.imshow(residuals_shifted, origin='lower')
    # ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('residuals re-centered')

    ax = axes[5]
    im = ax.imshow(correction, origin='lower')
    # ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('corrections on previous PSF')

    ax = axes[6]
    im = ax.imshow(np.log10(psf_new), origin='lower', vmin=vmin, vmax=vmax)
    # ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('new proposed PSF')
    return fig
