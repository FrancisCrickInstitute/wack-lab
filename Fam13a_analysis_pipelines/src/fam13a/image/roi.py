from typing import Iterable, Union

import numpy as np
import scipy as sp
import skimage

from . import consts


def circle(
    image: np.ndarray, hough_radii: Iterable[int] = (consts.HBEC_ROI_RADIUS, )
) -> np.ndarray:
    """Extract a circular ROI from an image using the Sobel operator and Hough
        transform

    Args:
        image: 2D numpy array of the image to extract the ROI from
        hough_radii: radii at which to search for the ROI

    Returns:
        boolean numpy array of the same shape as the input image
    """
    # detect edges in the image using the sobel operator
    edges = skimage.filters.sobel(image)

    # apply the circular Hough transform at all the specified radii
    hough_trans = skimage.transform.hough_circle(edges, hough_radii)
    # extract the parameters of the most prominent circle identified
    _, cx, cy, radii = skimage.transform.hough_circle_peaks(hough_trans,
                                                            hough_radii,
                                                            total_num_peaks=1)
    cx, cy, radii = cx[0], cy[0], radii[0]
    # define an expanded ROI and draw the circle on it
    # use expanded ROI to ensure complete circle is drawn on the image
    expanded_shape = [s+(radii*2) for s in image.shape]
    cy, cx = cy + radii, cx + radii
    roi = np.zeros(expanded_shape, dtype=bool)
    circy, circx = skimage.draw.circle_perimeter(cy, cx, radii,
                                                 shape=expanded_shape)
    roi[circy, circx] = True
    # fill in the circle in the ROI so we get a complete mask, rather than a bounding circumference
    roi = sp.ndimage.binary_fill_holes(roi)
    # crop out the expanded region
    roi = roi[radii:-radii, radii:-radii]

    return roi
