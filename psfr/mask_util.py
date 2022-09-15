import numpy as np


def mask_configuration(star_list, mask_list=None, saturation_limit=None):
    """
    configures the fitting masks for individual stars based on an optional input mask list and saturation limits

    Parameters
    ----------
    star_list : list of 2d numpy arrays
        list of cutout stars (to be included in the fitting)
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included
    saturation_limit: None, float or list of floats with same length as star_list
        saturation limit of the pixels. Pixel values at and above this value will be masked

    Returns
    -------
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        an updated list of masks for each individual star taking into account saturation
    use_mask : boolean
        if True, indicates that the masks need to be used, otherwise calculations will ignore mask for speed-ups
    """
    if saturation_limit is None and mask_list is None:
        use_mask = False
    else:
        use_mask = True
    n_star = len(star_list)
    # define saturation masks
    if saturation_limit is not None:
        n_sat = len(np.atleast_1d(saturation_limit))
        if n_sat == 1:
            saturation_limit = [saturation_limit] * n_star
        else:
            if n_sat != n_star:
                raise ValueError('saturation_limit is a list with non-equal length to star_list.')
    # initiate a mask_list that accepts all pixels
    if mask_list is None:
        mask_list = []
        for i, star in enumerate(star_list):
            mask = np.ones_like(star)
            mask_list.append(mask)
    # add threshold for saturation in mask
    if saturation_limit is not None:
        for i, mask in enumerate(mask_list):
            mask[star_list[i] > saturation_limit[i]] = 0
    return mask_list, use_mask
