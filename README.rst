===========================================
PSFr - Point Spread Function reconstruction
===========================================

|PyPI| |CI| |Docs| |Coveralls|

.. |CI| image:: https://github.com/sibirrer/psfr/workflows/Tests/badge.svg
    :target: https://github.com/sibirrer/psfr/actions

.. |Docs| image:: https://readthedocs.org/projects/psfr/badge/?version=latest
        :target: http://psfr.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. |Coveralls| image:: https://coveralls.io/repos/github/sibirrer/psfr/badge.svg?branch=main
    :target: https://coveralls.io/github/sibirrer/psfr?branch=main

.. |PyPI| image:: https://img.shields.io/pypi/v/psfr?label=PyPI&logo=pypi
    :target: https://pypi.python.org/pypi/psfr

.. image:: https://github.com/sibirrer/psfr/blob/main/docs/_static/stacked_psf_animation.gif

Point Spread Function reconstruction for astronomical
ground- and space-based imaging data.


Example
-------

.. code-block:: python

    # get cutout stars in the field of a JWST observation (example import)
    from psfr.util import jwst_example_stars
    star_list_jwst = jwst_example_stars()

    # run PSF reconstruction (see documentation for further options)
    from psfr.psfr import stack_psf
    psf_model, center_list, mask_list = stack_psf(star_list_jwst, oversampling=4,
                                                  saturation_limit=None, num_iteration=50)

We further refer to the example Notebook_ and the Documentation_.

.. _Notebook: https://github.com/sibirrer/psfr/blob/main/notebooks/JWST_PSF_reconstruction.ipynb
.. _Documentation: https://psfr.readthedocs.io/en/latest/


Features
--------

* Iterative PSF reconstruction given cutouts of individual stars or other point-like sources.
* Sub-pixel astrometric shifts calculated and accounted for while performing the PSF reconstruction.
* PSF reconstruction available in super-sampling resolution.
* Masking pixels, saturation levels and other options to deal with artifacts in the data.

Algorithm
---------
The algorithm to iteratively propose a (optionally oversampled) PSF from a set of star cutouts goes as follow:

(1) Find centroid pixel of all the stars

(2) Stack all the stars for an initial guess of the PSF (ignoring sub-pixel offsets)

(3) Fit the subpixel centroid with the PSF model estimate

(4) Shift PSF with sub-pixel interpolation to the sub-pixel position of individual stars

(5) Retrieve residuals of the shifted PSF model relative to the data of the cutouts

(6) Apply an inverse sub-pixel shift of the residuals to be focused on the center of the pixel

(7) Based on teh inverse shifted residuals of a set of fixed stars, propose a correction to the previous PSF model

(8) Repeat step (4) - (7) multiple times with the option to repeat step (3)


Details and options for the different steps can be found in the documentation and the source code.


Used by
-------
PSFr is in use with James Webb Space Telescope imaging data (i.e., `Santini et al. 2022  <https://ui.adsabs.harvard.edu/abs/2022arXiv220711379S/abstract>`_,
`Merlin et al. 2022  <https://ui.adsabs.harvard.edu/abs/2022arXiv220711701M/abstract>`_,
`Yang et al. 2022  <https://ui.adsabs.harvard.edu/abs/2022arXiv220713101Y/abstract>`_).
The iterative PSF reconstruction procedure was originally developed and used for analyzing strongly lensed quasars
(i.e., `Birrer et al. 2019 <https://ui.adsabs.harvard.edu/#abs/2018arXiv180901274B/abstract>`_
, `Shajib et al. 2018 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.5649S>`_ ,
`Shajib et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019arXiv191006306S/abstract>`_ ,
`Schmidt et al. 2022 <https://arxiv.org/abs/2206.04696>`_).


Other resources
---------------

We also refer to the astropy core package
`photutils <https://photutils.readthedocs.io/en/stable/index.html>`_
and in particular to the empirical PSF module
`ePSF <https://photutils.readthedocs.io/en/stable/epsf.html#build-epsf>`_ .


Credits
-------

The software is an off-spring of the iterative PSF reconstruction scheme of `lenstronomy <https://github.com/lenstronomy/lenstronomy>`_, in particular its `psf_fitting.py <https://github.com/lenstronomy/lenstronomy/blob/v1.10.4/lenstronomy/Workflow/psf_fitting.py>`_ functionalities.

If you make use of this software, please cite: 'This code is using PSFr (Birrer et al. in prep) utilizing features of
lenstronomy (`Birrer et al. 2021 <https://joss.theoj.org/papers/10.21105/joss.03283>`_)'.
