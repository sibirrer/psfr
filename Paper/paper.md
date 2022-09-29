---
title: 'PSFr - Point Spread Function reconstruction'
tags:
  - Python
  - astronomy
  - point spread function reconstruction
authors:
  - name: Simon Birrer
    orcid: 0000-0003-3195-5507
  - corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Vikram Bhamre
    # orcid: 0000-0000-0000-0000
    affiliation: 4 
  - name: Anna Nierenberg
    orcid: 0000-0001-6809-2536
    affiliation: 5 
  - name: Lilan Yang
    orcid: 0000-0002-8434-880X
    affiliation: 6 
affiliations:
  - name: Kavli Institute for Particle Astrophysics and Cosmology and Department of Physics, Stanford University, Stanford, CA 94305, USA
    index: 1
  - name: SLAC National Accelerator Laboratory, Menlo Park, CA, 94025, USA
    index: 2
  - name: Department of Physics and Astronomy, Stony Brook University, Stony Brook, NY 11794, USA
    index: 3
  - name: Highschool
    index: 4
  - name: University of California Merced, Department of Physics 5200 North Lake Rd. Merced, CA 9534, USA
    index 5
  - name: Kavli Institute for the Physics and Mathematics of the Universe, The University of Tokyo, Kashiwa, Japan 277-8583
    index 6

date: 01 October 2022
codeRepository: https://github.com/sibirrer/psfr
license: BSD-3 Clause
bibliography: paper.bib
---

# Summary

`PSFr` is a Python package to empirically reconstruct an oversampled version of the point spread function (PSF) from 
astronomical imaging observations.
`PSFr` provides a light-weighted API of a refined version of an algorithm originally implemented in `lenstronomy` [@lenstronomyI; @lenstronomyII].
`PSFr` provides user support with different artifacts in the data and supports the masking of pixels, or the treatment of saturation levels.
`PSFr` has been successfully used to reconstruct the PSF from multiply imaged lensed quasar images observed by the Hubble Space Telescope
in a crowded lensing environment and more recently with James Webb Space Telescope (JWST) imaging data for a wide dynamical flux range.



# Statement of need

The point spread function (PSF) of astronomical imaging data is dominated by diffraction in space-based observatories, 
and by the athomspheric distortions of the wavefront for ground-based observatories.
The characterization of the PSF at and below the pixel level of the data
is a key component to extract accurate and precise information from telescope facilities on the pixel-level data.
For example, the PSF model is crucial in providing high-precisino positional (astrometric) information of objects at 
sub-pixel scales, to de-blend different astronomical sources and their respective fluxes, 
or to describe the intrinsic shape and morphologies of galaxies.

The PSF is often time-varying, wavelenght and color dependent, and changes with the position on the focal point of the 
instrument. The most accurate and precise models of the PSF could often be derived empirically from the same or 
near-identical observations in time and space. In particular, objects in the imaging data that are bright and known to 
be point-like provide a realization of the sampling of the PSF [@AndersenKing:2000]. 
The combination and interpolation between several point-like objects in imaging data allows us to gain information 
about the PSF below the pixel scale. Such techniques become ever more important with the current and upcoming 
diffraction-limited space-based observatories (such as the Hubble Space Telescope and the James Webb Space Telescope)
and the large-aperture ground based extremely large telescopes (ELT's) with extreme adaptive optics performance.

`PSFr` performs iterative PSF reconstruction given cutouts of individual stars or other point-like sources.
`PSFr` calculates and accounts for sub-pixel astrometric shifts while performing the PSF reconstruction and supports
the reconstruction in oversampled resolution compared to the data.
`PSFr` provides a light-weighted API of a refined version of an algorithm originally implemented in `lenstronomy` [@lenstronomyI; @lenstronomyII].
The API supports the masking of pixels, scoping with saturation in the CCD detectors and other options to deal with artifacts in the data.

The algorithm was first used to iteratively reconstruct the PSF in the modeling of doubly and quadruply lensed quasars 
from HST imaging data on HST imaging data [@Birrer:2017; @Birrer:2019; @Shajib:2019; @Shajib:2020; @Schmidt:2022]
and ground-based adaptive optics imaging [@Shajib:2021]. With first light of JWST, the method has been refined and has 
since been used for studies of galaxy evolution and quasar-host galaxy docomposition studies [@Santini:2022; @Merlin:2022; @Yang:2022; @Ding:2022].




# Algorithm

The algorithm to iteratively propose a (optionally oversampled) PSF from a set of star cutouts goes as follow:


(1) Stack all the stars for an initial guess of the PSF on the centroid pixel (ignoring sub-pixel offsets)

(2) Fit the subpixel centroid with the PSF model estimate

(3) Shift PSF with sub-pixel interpolation to the sub-pixel position of individual stars

(4) Retrieve residuals of the shifted PSF model relative to the data of the cutouts

(5) Apply an inverse sub-pixel shift of the residuals to be focused on the center of the pixel

(6) Based on teh inverse shifted residuals of a set of fixed stars, propose a correction to the previous PSF model

(7) Repeat step (3) - (6) multiple times with the option to repeat step (2)

Details and options for the different steps can be found in the documentation and the source code.

# Related open source software

- [`lenstronomy`](https://github.com/sibirrer/lenstronomy) [@lenstronomyI; @lenstronomyII]
- [`photutils`](https://github.com/astropy/photutils) [@photutils]
- [`WebbPSF`](https://github.com/spacetelescope/webbpsf) [@webbpsf]


# Acknowledgements

We acknowledge support from ...

# References