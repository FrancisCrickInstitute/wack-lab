from typing import Tuple

import numpy as np
from skimage.util import view_as_blocks
from tqdm import tqdm


def extract(
    array: np.ndarray, window: Tuple[int, ...],
    steps: Tuple[int, ...] = None, mode: str = 'valid'
) -> np.ndarray:
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.

    Args:
        array: Array to which the rolling window is applied.
        window: Either a single integer to create a window of only the last axis or a
            tuple to create it for the last len(window) axes. 0 can be used as a
            to ignore a dimension in the window.
        steps: step size of the window in each dimension
        mode: padding mode to use. If 'valid' (default) will only extract 
            patches the lie completely within the original image. Otherwise it 
            should be one of the mode options in 
            https://numpy.org/doc/stable/reference/generated/numpy.pad.html 

    Returns:
        A view on `array` which is smaller to fit the windows and has windows added
        dimensions (0s not counting), ie. every point of `array` is an array of size
        window.
    """
    if steps is None:
        steps = window

    if mode != 'valid':
        pad_width = [(p_size - s_size, )
                     for p_size, s_size in zip(window, steps)]
        array = np.pad(array, pad_width=pad_width, mode=mode)

    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError(
            "`window` length must be less or equal `array` dimension.")

    _steps = np.ones_like(orig_shape)
    steps = np.atleast_1d(steps)
    if steps.ndim != 1:
        raise ValueError("`steps` must be either a scalar or one dimensional.")
    if len(steps) > array.ndim:
        raise ValueError(
            "`steps` cannot be longer then the `array` dimension.")
    # does not enforce alignment, so that steps can be same as window too.
    _steps[-len(steps):] = steps

    if np.any(steps < 1):
        raise ValueError("All elements of `steps` must be larger then 1.")
    steps = _steps

    wsteps = np.ones_like(window)

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError(
            "`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + steps - 1) // steps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= steps
    new_strides = array.strides[-len(window):] * wsteps

    # The full new shape and strides:
    new_shape = np.concatenate((shape, window))
    new_strides = np.concatenate((strides, new_strides))

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    patched_arr = np.lib.stride_tricks.as_strided(
        array, shape=new_shape, strides=new_strides)
    patched_arr = patched_arr.reshape(-1, *window)

    return patched_arr


def merge(
    patches: np.ndarray, img_shape: Tuple[int, int],
    steps: Tuple[int, int] = None, padded: bool = False
) -> np.ndarray:
    """Merge a collection of patches into a single array with potential overlap

    Args:
        patches: array with shape (N, H_p, W_p) where N is the number of
            patches, and H_p, W_p are the height and width of the patches
            NOTE: merge assumes patches are provided in row-major order
        image_shape: tuple containing (H_i, W_i) where H_i, W_i are the 
            height and width of the orignal image
        steps: step size in each dimension taken when extracting the patches. If
            `None` then steps is set equal to the patch size
        padded: if False (default) assumes patches were extracted in "valid" 
            mode (i.e. no padding was done on patch extraction). If the image
            was padded during patch extraction this should be set to True.
    """
    # extract the image, step, and window height/width for convenience
    img_h, img_w = img_shape
    window_h, window_w = patches.shape[1:]

    if steps is None:
        step_h, step_w = window_h, window_w
    else:
        step_h, step_w = steps

    # construct the size of padding and padded image
    pad_h, pad_w = 0, 0
    if padded:
        pad_h, pad_w = window_h - step_h, window_w - step_w

    pad_img_h, pad_img_w = img_h + 2*pad_h, img_w + 2*pad_w

    # calculate number of patches per column and per row
    # this is needed to figure out each patch position within the full image
    pp_col = ((pad_img_w - window_w) // step_w) + 1
    pp_row = ((pad_img_h - window_h) // step_h) + 1

    if pp_col*pp_row != patches.shape[0]:
        raise ValueError(
            f'Invalid number of patches provided for the given image shape. Got {patches.shape[0]}, expected {pp_col*pp_row}'
        )

    # initialise an empty image to be filled in with the patches
    img = np.zeros((pad_img_h, pad_img_w))

    counter = np.zeros_like(img, dtype=np.float32)

    for idx, patch in enumerate(patches):
        # calculate the top-left corner position of the patch assuming
        # row-major order
        col_idx = idx % pp_col
        row_idx = (idx - col_idx) // pp_col

        # construct index slices for the patch
        row_start, row_end = (step_h * row_idx), (step_h * row_idx + window_h)
        col_start, col_end = (step_w * col_idx), (step_w * col_idx + window_w)

        # add the patch values in the correct positions
        img[row_start:row_end, col_start:col_end] += patch
        # add 1 to the counter for the positions covered by the current
        # patch, we want to keep track of number of times a each position is
        # seen so we can average the result at the end
        counter[row_start:row_end, col_start:col_end] += 1

    # crop out padded regions
    if padded:
        img = img[pad_h:-pad_h, pad_w:-pad_w]
        counter = counter[pad_h:-pad_h, pad_w:-pad_w]

    # average the array values
    img /= counter

    return img
