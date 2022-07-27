===========================================
PSFr - Point Spread Function reconstruction
===========================================

.. image:: https://github.com/sibirrer/psfr/workflows/Tests/badge.svg
    :target: https://github.com/sibirrer/psfr/actions

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
It is in use with James Webb Space Telescope imaging data.
The iterative PSF reconstruction procedure was also used for strongly lensed quasars,
i.e., `Birrer et al. 2019 <https://ui.adsabs.harvard.edu/#abs/2018arXiv180901274B/abstract>`_
, `Shajib et al. 2018 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.5649S>`_ ,
`Shajib et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019arXiv191006306S/abstract>`_,
 `Schmidt et al. 2022 <https://arxiv.org/abs/2206.04696>`_.

Credits
-------

The code is an off-spring of the iterative PSF reconstruction scheme of `lenstronomy <https://github.com/sibirrer/lenstronomy>`_
, in particular the `psf_fitting.py <https://github.com/sibirrer/lenstronomy/lenstronomy/Workflow/psf_fitting.py>`_ functionalities.

If you make use of this code, please cite: 'This code is using PSFr (Birrer et al. in prep) utilizing features of
lenstronomy (`Birrer et al. 2021 <https://joss.theoj.org/papers/10.21105/joss.03283>`_)'.
