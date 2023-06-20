import numpy as np
import scipy as sp


def filter_area(
    frame: np.ndarray, min_area: int = 0, max_area: int = np.inf
) -> np.ndarray:
    """Remove contiguous regions whos area is not in the given range

    Args:
        frame: image to filter
        min_area: minimum area, in pixels, for valid contiguous regions 
            (threshold is inclusive, i.e. area >= min_area)
        max_area: maximum area, in pixels, for valid contiguous regions
            (threshold is inclusive, i.e. area <= max_area)
    
    Returns:
        numpy array of the same shape as frame with objects not in the given 
        area range removed
    """
    # create a copy of the input frame so we don't modify the original variable
    frame = frame.copy()
    # extract all contiguous regions in the frame and their areas
    labelled, _ = sp.ndimage.label(frame)
    labels, areas = np.unique(labelled, return_counts=True)

    # validate each contiguous region and remove from the frame if it fails
    for label, area in zip(labels, areas):
        if area < min_area or area > max_area:
            frame[labelled == label] = 0

    return frame
