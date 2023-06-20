from typing import Tuple

import cv2
import numpy as np
import scipy as sp
from scipy import signal
from scipy import ndimage as ndi
import skimage
from skimage.segmentation import watershed
from skimage.morphology import disk

from . import consts, utils

def gradient(
    frame: np.ndarray, open_kernel: np.ndarray = disk(3),
    min_area: int = consts.GRAD_MIN_AREA
) -> np.ndarray:
    """Segment the embryos in a single image using the intensity gradient

    Args:
        frame: array, with shape (H, W), of a grayscale image
        open_kernel: square 2D array defining the kernel to use in the morph open operation
        min_area: mininum threshold value to use after initial segmentation. 
            Any objects whose area is less than this value are discarded from 
            the segmentation mask
    
    Returns:
        2D array, with shape (H, W), of the segmented mask based on the 
        original intensity gradients 
    """
    # apply thea 2D convolution using the Scharr kernel to pick out intensity
    # gradient magnitude and orientation
    mask = signal.convolve2d(frame, consts.SCHARR_KERNEL, boundary='symmetric',
                             mode='same')
    # gradient magnitudes are given by the absolute values
    mask = np.absolute(mask)

    # rescale image values and convert to dtype uint8 so we can use the
    # cv2.adaptiveThreshold function
    mask /= mask.max()
    mask *= 255
    mask = np.round(mask).astype(np.uint8)

    # apply an adaptive threshold to the gradient maginitudes
    # this draws quite a clean boundary around single embryos most of the time
    mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 consts.ADAPTIVE_THRESH_BLOCK_SIZE, 0)

    # the threshold operation leaves a lot of noise in the image, to clean
    # it up we apply a morphological opening and then remove any contiguous
    # segmented regions that have too small an area
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = utils.filter_area(mask, min_area=min_area)

    return mask

def run_segmentation(grad: np.ndarray,
                     markers: np.ndarray,
                     mask_er: np.ndarray,
                     compactness: float,
                     min_size: int,
                     max_size: int) -> Tuple[np.ndarray,
                                             np.ndarray,
                                             list]:
    """Run watershed segmentation based on grad with:
        - Markers as initial markers
        - Masked by mask_er
        - Compactness
        - Min and max size of segments

    Args:
        grad: array, with shape (H, W), used for segmentation
        markers: Initial markers for segmentation
        mask_er: Mask to use for segmentation
        compactness: Compactness parameter of watershed segmentation
        min_size: Minimum size of segment
        max_size: Maximum size of segment
        markers: Markers to start watershed segmentation from
    
    Returns:
        2D array, with shape (H, W) with segmentation,
        2D array, with shape (H, W) and centres of inertia of segments,
        List with failed markers
    """
    
    # Apply watershed over mask_er with markers as initial points
    segm = watershed(grad, markers=markers, compactness=compactness, mask=mask_er)
    
    # Remove small areas that are not segmented
    segm = skimage.morphology.area_closing(segm, area_threshold=500)
    
    # Remove segments at border
    segm_cleared = skimage.segmentation.clear_border(segm)
    
    # Initialize
    new_markers = np.zeros(segm.shape)
    
    # Convert to int
    markers = markers.astype(int)
    
    # Initialize
    failed_markers = []
    
    # Go through all segments and calculate new_markers
    for j in range(1, np.max(markers)+1):
        seg_bool = segm == j
        # mark segment as failed if size is empty
        if np.sum(seg_bool) == 0:
            new_markers = new_markers + (markers == j)*j
            failed_markers.append(j)
        # mark segment as failed if size is too large or too small 
        # and remove segment from output
        elif (np.sum(seg_bool) < min_size or np.sum(seg_bool) > max_size):
            segm[seg_bool] = 0
            new_markers = new_markers + (markers == j)*j
            failed_markers.append(j)
        # If segment size is within expected limits
        else:
            # If segment has not been cleared by clearing bordering segments
            if j in np.unique(segm_cleared):
                x, y = sp.ndimage.center_of_mass(seg_bool)
                new_markers[int(x), int(y)] = j
    
    new_markers = new_markers.astype(int)
    return segm, new_markers, failed_markers

def process_markers(frame: np.ndarray,
                    min_size: int,
                    max_size: int,
                    markers: np.ndarray) -> np.ndarray:
    
    """Segment frame with initial markers with segments limited by min and max size

    Args:
        frame: array, with shape (H, W), of a grayscale image
        min_size: Minimum size of segment
        max_size: Maximum size of segment
        markers: Markers to start watershed segmentation from
    
    Returns:
        2D array, with shape (H, W) and centres of inertia of segments
    """
    
    # Calculate mask and apply dilation
    mask = cv2.inRange(frame, np.array([12, 60, 42]), np.array([255, 255, 255]))
    mask_er = skimage.morphology.binary_erosion(mask, skimage.morphology.disk(5))
    
    # Calculate gradient with Scharr operator
    grad = skimage.filters.scharr(frame[:,:,1])

    # Segment with a small value of compactness first.
    # Then, repeat segmentation for the failed markers
    # with a higher value of compactness.
    #
    # TODO: Write this into a for/while loop
    compactness = 0
    segm_1, new_markers_1, failed_markers_1 = run_segmentation(grad,
                                                               markers,
                                                               mask_er,
                                                               compactness,
                                                               min_size,
                                                               max_size)
    # Assign outputs to final function outputs
    filtered_failed_markers_1 = failed_markers_1
    segm = segm_1
    new_markers = new_markers_1
    
    if len(filtered_failed_markers_1) > 0: # If remaining failed segments
        # Run segmentation
        compactness = 0.0002
        segm_2, new_markers_2, failed_markers_2 = run_segmentation(grad,
                                                                   markers,
                                                                   mask_er,
                                                                   compactness,
                                                                   min_size,
                                                                   max_size)
        # Update outputs for failed segments
        for failed_marker in filtered_failed_markers_1:
            segm[segm_2 == failed_marker] = segm_2[segm_2 == failed_marker]
            new_markers[new_markers == failed_marker] = new_markers_2[new_markers == failed_marker]
            new_markers[new_markers_2 == failed_marker] = new_markers_2[new_markers_2 == failed_marker]
        filtered_failed_markers_2 = [x for x in failed_markers_2 if x in filtered_failed_markers_1]
        
        if len(filtered_failed_markers_2) > 0: # If remaining failed segments
            # Run segmentation
            compactness = 0.0004
            segm_3, new_markers_3, failed_markers_3 = run_segmentation(grad,
                                                                       markers,
                                                                       mask_er,
                                                                       compactness,
                                                                       min_size,
                                                                       max_size)
            # Update outputs for failed segments
            for failed_marker in filtered_failed_markers_2:
                segm[segm_3 == failed_marker] = segm_3[segm_3 == failed_marker]
                new_markers[new_markers == failed_marker] = new_markers_3[new_markers == failed_marker]
                new_markers[new_markers_3 == failed_marker] = new_markers_3[new_markers_3 == failed_marker]
    
    # Renumber segments
    i = 1
    for j in np.unique(new_markers):
        if j != 0:
            seg_bool = new_markers == j
            new_markers[seg_bool] = i
            i = i + 1
    
    return new_markers
