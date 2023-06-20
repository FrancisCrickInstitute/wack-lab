from functools import partial
from multiprocessing import Pool
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from . import consts, patch


def estimate(
    frames: np.ndarray,
    kernel: Tuple[int, int] = consts.XENOPUS_BCKGR_GAUS_KERNEL,
    percentile: float = consts.XENOPUS_BCKGR_PERCENTILE
) -> np.ndarray:
    """Estimate the background of collection of frames

    Args:
        frames: numpy array containing grayscale images with shape (N, H, W).
             Where N is the frame count, and H,W are the height and width of 
             the frames. NOTE algorithm assumes frame values are 0-255
        kernel: tuple of length 2 (H_k, W_k) giving the size of the 
            kernel to use in the Gaussian filter
        percentile: float between 0-1 indicating the perncetile to use in 
            background estimation 

    Returns:
        numpy array with shape (H, W) containing the background estimate
    """
    # estimate background per-pixel over all frames
    bkg = np.percentile(frames, q=percentile, axis=0)
    # convert estimate to correct dtype
    bkg = bkg.round().astype(np.uint8)

    # smooth estimate using a Gaussian kernel
    bkg = cv2.GaussianBlur(bkg, kernel, 0)

    return bkg


def remove(frames: np.ndarray, invert: bool = False) -> np.ndarray:
    """Remove the background from a collection of frames

    NOTE: algorithm assumes the background is static. It also assumes the 
        background has a lower instensity (is darker) than the foreground, if 
        this is not the case set the `invert` flag to `True`

    Args:
        frames: numpy array containing grayscale images with shape (N, H, W).
             Where N is the frame count, and H,W are the height and width of 
             the frames. NOTE algorithm assumes frame values are 0-255
        invert: flag indicating wether or not to invert the image intensity

    Returns:
        numpy array of the frames with the background removed
    """

    # to invert colours subtract pixel values from max possible value
    if invert:
        frames = np.iinfo(np.uint8).max - frames

    # estimate background
    bkg = estimate(frames)

    # subtract background from all frames
    # need to convert frames to int16 to be able to handle cases where
    # background intensity is higher than foreground
    frames = frames.astype(np.int16) - bkg

    # lower bound the frames at 0 and convert back to uint8
    frames = np.maximum(frames, 0).astype(np.uint8)

    # if colours where inverted, undo the change after removing the background
    if invert:
        frames = np.iinfo(np.uint8).max - frames

    return frames


def estimate_patched(
    frame: np.ndarray, patch_size: Tuple[int, int]
) -> np.ndarray:
    """Estimate the background of an image using sub-regions

    NOTE: algorithm assumes the  background has a lower instensity (is darker)
        than the foreground, if this is not the case set the `invert` flag to
        `True`

    Args:
        frame: numpy array of a single frame
        patch_size: (H, W) of the subregions to estimate the background in

    Returns:
        numpy array of the same shape as the input frame containing the 
        estimate background
    """

    # extract the patches from the frame
    patches = patch.extract(frame, patch_size)
    # flatten each patch, as we no longer care about spatial information within
    # a patch, we only need to know which patch a particular pixel is in
    patches = patches.reshape(patches.shape[0], -1)
    # for each patch estimate background by taking the `XENOPUS_BCKGR_PERCENTILE`
    # darkest pixel in the region
    bkg_values = np.percentile(patches, q=consts.XENOPUS_BCKGR_PERCENTILE,
                               axis=-1)

    # construct background patches as solid blocks of the extract value
    # and stack them back into a single array (N, H, W)
    bkg_patches = [np.full(patch_size, val, dtype=np.uint8)
                   for val in bkg_values]
    bkg_patches = np.stack(bkg_patches, axis=0)

    # merge the patches back into an array with the same shape as the original frame
    bkg = patch.merge(bkg_patches, frame.shape)

    # want to apply a Gaussian blur to the background to smooth out the
    # boundaires, to do this we want to use a kernel the same size as an
    # individual patch. Also need to ensure kernel size is odd-valued
    g_kernel = []
    for p_dim in patch_size:
        if p_dim % 2 == 0:
            g_kernel.append(p_dim+1)
        else:
            g_kernel.append(p_dim)

    bkg = cv2.GaussianBlur(bkg, tuple(g_kernel), 0)

    return bkg


def remove_patched_single(
    frame: np.ndarray, patch_size: Tuple[int, int], invert: bool = False
) -> np.ndarray:

    if invert:
        frame = np.iinfo(np.uint8).max - frame

    # estimate background for each frame
    bkg = estimate_patched(frame, patch_size).astype(np.uint8)

    # need to convert frame to int16 to be able to handle cases where
    # background intensity is higher than foreground
    frame = frame.astype(np.int16)
    # subtract backgrounds from all frames
    frame -= bkg

    # lower bound the frames at 0 and convert back to uint8
    frame = np.maximum(frame, 0).astype(np.uint8)

    # if colours where inverted, undo the change after removing the background
    if invert:
        frame = np.iinfo(np.uint8).max - frame

    return frame


def remove_patched(
    frames: np.ndarray, patch_size: Tuple[int, int], invert: bool = False,
    ncpus: int = 1
) -> np.ndarray:
    """Remove the background from a collection of frames using subregions per 
        frame

    Args:
        frames: numpy array containing grayscale images with shape (N, H, W).
             Where N is the frame count, and H,W are the height and width of 
             the frames. NOTE algorithm assumes frame values are 0-255
        patch_size: (H, W) of the subregions to estimate the background in
        invert: flag indicating wether or not to invert the image intensity

    Returns:
        numpy array of the frames with the background removed

    """
    clean_frames = []
    frames = [frame for frame in frames]

    if ncpus > 1:
        par_func = partial(remove_patched_single,
                           patch_size=patch_size, invert=invert)

        with Pool(ncpus) as p:
            for frame in tqdm(p.imap(par_func, frames)):
                clean_frames.append(frame)

    else:
        for frame in tqdm(frames):
            frame = remove_patched_single(frame, patch_size, invert)
            clean_frames.append(frame)

    clean_frames = np.stack(clean_frames, axis=0)

    return clean_frames
