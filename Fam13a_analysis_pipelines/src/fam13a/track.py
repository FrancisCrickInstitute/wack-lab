from typing import Iterable, Tuple

import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import euclidean_distances

from . import consts


def euclidean_dist(prev_pts: np.ndarray, curr_pts: np.ndarray) -> np.ndarray:
    """Calculates the euclidian distance between all possible pairs of points

    Args:
        prev_pts: (N, D) array where N is the number of points and D is the
            dimensionality of the points
        curr_pts: (M, D) array where M is the number of points and D is the
            dimensionality of the points. M and N do not have to be the same, 
            but D must be consistent

    Returns:
        (N, M) array of all pairwise euclidean distances between points
    """

    if prev_pts.ndim == 1:
        prev_pts = np.expand_dims(prev_pts, axis=0)
    if curr_pts.ndim == 1:
        curr_pts = np.expand_dims(curr_pts, axis=0)

    return euclidean_distances(prev_pts, curr_pts)


def l2_dist(vc1: Iterable[float], vc2: Iterable[float]) -> float:
    """Calculate the L2 norm between 2 vectors

    Args:
        vc1: numeric iterable of the first vector
        vc2: numeric iterable of the second vector of the same length as vc1

    Returns:
        the L2-norm distance between the 2 vectors
    """

    if len(vc1) != len(vc2):
        raise ValueError(
            f'vectors must have the same length. Instead got: {vc1} vs {vc2}'
        )

    return sum((vc1_coord - vc2_coord)**2 for vc1_coord, vc2_coord in zip(vc1, vc2))**0.5


def align_points(
    prev_pts: np.ndarray, curr_pts: np.ndarray,
    move_thr: float = consts.FRAME_MOVE_THRESHOLD,
    return_missing: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies the Munkres algorithm to align 2 sets of points

    Args:
        prev_pts: (N, 1, D) array where N is the number of points and D is the
            dimensionality of the points
        curr_pts: (M, 1, D) array where M is the number of points and D is the
            dimensionality of the points. M and N do not have to be the same, 
            but D must be consistent
        move_thr: upper limit (in pixels) on the distance a point can move between 2 
            sequential frames. Aligned points above this limit are relabelled 
            as new points

    Returns:
        a tuple of 3 arrays:
            - shape (K, D) of previous points that have been aligned, where K 
                is the number of aligned points. NOTE K <= min(M,N)
            - shape (K, D) of current points that have been aligned. Ordering 
                is the same as the aligned previous points array
            - shape (M - K, D) of new points from curr_pts which were not aligned.
    """

    # drop dimensions of size 1
    prev_pts, curr_pts = np.squeeze(prev_pts), np.squeeze(curr_pts)
    # handle edge case where only 1 point is available
    if prev_pts.ndim == 1:
        prev_pts = np.expand_dims(prev_pts, axis=0)
    if curr_pts.ndim == 1:
        curr_pts = np.expand_dims(curr_pts, axis=0)

    # handle cases when no points are present
    if curr_pts.shape[0] == 0:
        if return_missing:
            return [np.empty(shape=(0, 2)), np.empty(shape=(0, 2)),
                    np.empty(shape=(0, 2)), prev_pts]
        else:
            return [np.empty(shape=(0, 2))] * 3
    if prev_pts.shape[0] == 0:
        if return_missing:
            return [np.empty(shape=(0, 2)), np.empty(shape=(0, 2)), curr_pts,
                    np.empty(shape=(0, 2))]
        else:
            return [np.empty(shape=(0, 2)), np.empty(shape=(0, 2)), curr_pts]

    # construct an all vs all cost matrix, using the euclidean distance as the metric
    cost_mat = euclidean_dist(prev_pts, curr_pts)

    # use Munkres algorithm to align the previous set of points with the current ones
    prev_pts_idxs, curr_pts_idxs = sp.optimize.linear_sum_assignment(cost_mat)

    # drop alignemt of points which have moved too far
    cost_row = cost_mat[prev_pts_idxs, curr_pts_idxs]
    filter_alignments = np.where(cost_row < move_thr)

    prev_pts_idxs, curr_pts_idxs = prev_pts_idxs[filter_alignments], curr_pts_idxs[filter_alignments]

    # extract the aligned points
    aligned_prev_pts = prev_pts[prev_pts_idxs, ...]
    aligned_curr_pts = curr_pts[curr_pts_idxs, ...]
    new_pts = np.delete(curr_pts, curr_pts_idxs, axis=0)

    if return_missing:
        missing_pts = np.delete(prev_pts, prev_pts_idxs, axis=0)
        output = (aligned_prev_pts, aligned_curr_pts, new_pts, missing_pts)
    else:
        output = (aligned_prev_pts, aligned_curr_pts, new_pts)

    return output
