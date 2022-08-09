===========================================
PSFr - Point Spread Function reconstruction
===========================================

.. image:: https://github.com/sibirrer/psfr/workflows/Tests/badge.svg
    :target: https://github.com/sibirrer/psfr/actions

.. image:: https://github.com/sibirrer/psfr/blob/main/docs/_static/stacked_psf_animation.gif

Point Spread Function reconstruction for astronomical
ground- and space-based imaging data.

Animation or graphics
Link to example notebooks
Link to documentation


Features
--------

* Iterative PSF reconstruction given cutouts of individual stars or other point-like sources.
* Sub-pixel astrometric shifts calculated and accounted for on the fly.
* PSF reconstruction available in super-sampling resolution
* Masking pixels, saturation levels and other options.

Used by
-------
PSFr is in use with James Webb Space Telescope imaging data (`Santini et al. 2022  <https://ui.adsabs.harvard.edu/abs/2022arXiv220711379S/abstract>`_,
`Merlin et al. 2022  <https://ui.adsabs.harvard.edu/abs/2022arXiv220711701M/abstract>`_,
`Yang et al. 2022  <https://ui.adsabs.harvard.edu/abs/2022arXiv220713101Y/abstract>`_).
The iterative PSF reconstruction procedure was originally developed and used for analyzing strongly lensed quasars (
i.e., `Birrer et al. 2019 <https://ui.adsabs.harvard.edu/#abs/2018arXiv180901274B/abstract>`_
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

The software is an off-spring of the iterative PSF reconstruction scheme of `lenstronomy <https://github.com/sibirrer/lenstronomy>`_
, in particular its `psf_fitting.py <https://github.com/sibirrer/lenstronomy/lenstronomy/Workflow/psf_fitting.py>`_ functionalities.

If you make use of this software, please cite: 'This code is using PSFr (Birrer et al. in prep) utilizing features of
lenstronomy (`Birrer et al. 2021 <https://joss.theoj.org/papers/10.21105/joss.03283>`_)'.
