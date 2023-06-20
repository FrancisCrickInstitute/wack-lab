from typing import Dict, Tuple

import numpy as np
import scipy as sp


def centers_of_mass(frame: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """Calculate the center of mass for each object in the image:

    Args:
        frame: numpy array, shape (H, W), with objects in the image identified 
            by unique integer labes

    Returns:
        dictionary where the keys are the object integer labels, and the values
        are tuples of (x, y) position of the center of mass
    """
    coms = dict()

    # drop first value, as it is background
    for lab in np.unique(frame)[1:]:
        # calculate center of mass for each object in frame
        x, y = sp.ndimage.center_of_mass(frame == lab)
        # convert coordinates to integer values, so they can be used as indices
        # in the array
        coms[lab] = (int(x), int(y))

    return coms
