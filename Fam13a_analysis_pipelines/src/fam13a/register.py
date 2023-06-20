from typing import Tuple, Dict

import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm
from multiprocessing import Pool
import math

from .image import patch
from .track import l2_dist


def _start_idxs(arr: np.ndarray, size: Tuple[int, ...]) -> Tuple[int, ...]:
    # ensure pattern dimensionality equals arr dimensionality
    if arr.ndim != len(size):
        raise ValueError(
            f'arr and size must have the same dimensionality. Instead got {arr.ndim} vs {len(size)}'
        )

    dim_diffs = [a_dim - p_dim for a_dim, p_dim in zip(arr.shape, size)]

    # ensure the pattern size fits in the array
    pattern_fits = (diff >= 0 for diff in dim_diffs)
    if not all(pattern_fits):
        raise ValueError(
            f'Cannot extract pattern of size {size} from array with shape {arr.shape}'
        )

    # check the pattern can be extracted from exactly the center of arr
    are_central = (diff % 2 == 0 for diff in dim_diffs)
    if not all(are_central):
        raise ValueError(
            f'Cannot place pattern of size {size} in center of arr with shape {arr.shape}'
        )

    # construct the slices to extract the pattern
    start_idxs = tuple(diff // 2 for diff in dim_diffs)
    return start_idxs


def _find_rmse(arr: np.ndarray, ptrn: np.ndarray) -> Tuple[Tuple[int, ...], np.ndarray]:
    """Find the position in arr with the closest match to pattern based on 
    RMSE. Additionally all RMSE values are returned.

    Args:
        arr: N-dim array in which to search
        ptrn: N-dim array defining the pattern to search for. It must have 
            the same dimensionality as arr 

    Returns:
        tuple of integers, of length N, giving the position of best match to
            the pattern.
        1D array of RMSE values, of length M, where M is the number of possible 
            matches of ptrn inside arr.

    Example:
        if the ptrn shape is (3,3) and the arr shape is (5,5), then there are 
            (5-3+1)*(5-3+1)=9 possible matches for ptrn inside arr. Therefore 
            the length of the array of RSME values would be M=9.
    """
    if arr.ndim != ptrn.ndim:
        raise ValueError(
            f'arr and ptrn must have the same dimensionality. Instead got {arr.ndim} vs {ptrn.ndim}'
        )

    # extract all potential positions of the pattern in the array
    # only valid positions are considered i.e. positions where the pattern lies
    # entirely within arr
    potentials = patch.extract(arr, ptrn.shape, (1, 1)).astype(np.float64)

    # subtract the pattern from each potential position to make the search a
    # minimisation problem
    potentials -= ptrn

    # calculate the RMSE for each possible position
    potentials = potentials.reshape(potentials.shape[0], -1)
    rmse = np.sqrt(np.square(potentials).mean(axis=-1))
    val = rmse.min()

    # Using RMSE to estimate the movement between time-steps will always return
    # some position in the next time-step. However this is not always
    # desireable as it is entirely possible that the current search region
    # contains only noise. In this scenario we want to estimate a velocity of 0
    # but the min-RMSE value is not garuanteed to return that result. Instead
    # we apply a crude confidence threshold on the min-value before accepting
    # the new position
    if val < arr.mean():
        min_pos = np.argmin(rmse)
        # the rmse array is 1D, but the input arr and pattern can be any dimension
        # so we want to convert from a position in a 1D array to its N-dimensional
        # equivalent (assuming row-major ordering)
        unraveled_shape = [a_dim - p_dim + 1 for a_dim,
                           p_dim in zip(arr.shape, ptrn.shape)]
        idxs = np.unravel_index(min_pos, unraveled_shape)
    else:
        # return original position if confidence threshold not met
        # i.e. velocity of 0 in all directions
        idxs = _start_idxs(arr, ptrn.shape)

    return idxs, rmse


def _find_corr(arr: np.ndarray, ptrn: np.ndarray) -> Tuple[Tuple[int, ...], np.ndarray]:
    """Find the position in arr with the closest match to pattern based on 
    correlation. Additionally all correlation values are returned.

    Args:
        arr: N-dim array in which to search
        ptrn: N-dim array defining the pattern to search for. It must have 
            the same dimensionality as arr 

    Returns:
        tuple of integers, of length N, giving the position of best match to
            the pattern.
        1D array of correlation values, of length M, where M is the number of 
            possible matches of ptrn inside arr.
    """
    arr, ptrn = arr.astype(np.float64), ptrn.astype(np.float64)
    arr -= ptrn.mean()
    ptrn -= ptrn.mean()

    corrs = convolve2d(ptrn, np.flip(arr), mode='valid')

    # extract the position of the best match (highest correlation)
    max_pos = np.argmax(corrs)

    # max position is given from a 1D view
    # so we want to convert from a position in a 1D array to its N-dimensional
    # equivalent (assuming row-major ordering)
    idxs = np.unravel_index(max_pos, corrs.shape)

    return idxs, corrs


def find(arr: np.ndarray, ptrn: np.ndarray, method='rmse') -> Tuple[Tuple[int, ...], np.ndarray]:
    """Find the position in arr with the closest match to pattern. Also return the array of 
    errors that were produced for the different matches of the pattern in the array.

    Args:
        arr: N-dim array in which to search
        ptrn: N-dim array defining the pattern to search for. It must have 
            the same dimensionality as arr 

    Returns:
        tuple of integers, of length N, giving the position of best match to
            the pattern.
        1D array of errors, of length M, where each error corresponds to a 
            possible match of the pattern in the array. M is the number of 
            possible matches of the pattern in the array.
    """
    if arr.ndim != ptrn.ndim:
        raise ValueError(
            f'arr and ptrn must have the same dimensionality. Instead got {arr.ndim} vs {ptrn.ndim}'
        )

    method = method.lower()
    if method == 'rmse':
        idxs, errors = _find_rmse(arr, ptrn)
    elif method == 'corr':
        idxs, errors = _find_corr(arr, ptrn)
    else:
        raise NotImplementedError('search method {} not implemented')

    return idxs, errors


def pattern(arr: np.ndarray, size: Tuple[int, ...]) -> np.ndarray:
    """Extract the pattern in arr of the given size from the center of arr

    The pattern size must be set such that it can be placed exactly in the 
    center of arr.

    Args:
        arr: N-dim array to extract the pattern from
        size: dimensionality of the pattern to extract. Must have N entries

    Returns:
        numpy array containing the central slice of arr with shape of size
    """

    # construct the slices to extract the pattern
    start_idxs = _start_idxs(arr, size)
    end_idxs = [start_idx + p_dim for start_idx,
                p_dim in zip(start_idxs, size)]
    slices = tuple(slice(start_idx, end_idx)
                   for start_idx, end_idx in zip(start_idxs, end_idxs))

    return arr[slices]


def _estimate_shift(
    curr_region: np.ndarray, next_region: np.ndarray,
    pattern_size: Tuple[int, ...], dtype: type = np.int16
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the shift of the central block of pixels between the current
    region and the next region. Also return the errors of the possible matches
    of the central block in the next_region. 

    Args:
        curr_region: array containing the search region for the current time
            step. The search pattern will be extracted from the central values
            of this array 
        next_region: array containing the same search region in the next time 
            step. The pattern extracted from curr_region will be searched for 
            in this array. Must have the same dimensions as curr_region
        pattern_size: size of the pattern to extract from curr_region and 
            search for in next_region. Must have a value for each dimension in
            curr_region
        dtype: the data-type of the returned array

    Returns:
        1D array of the estimated shift of the pattern in each dimension. 
            Length of the array is equal to length of pattern_size.
        1D array of errors of the possible matches of the central block in 
            the next region. Length of the array is the number of possibile positions
            of the central block in the next_region. 
    """

    ptrn = pattern(curr_region, pattern_size)
    # calculate the index position of the extracted pattern
    start_idxs, _ = find(curr_region, ptrn)
    # find the best position of the pattern in the next region
    # also return the errors of the possible matches of ptrn in next_region
    end_idxs, errors = find(next_region, ptrn)

    # evaluate the estimated shift in position and return it as a 1D array
    shift = (end-start for start, end in zip(start_idxs, end_idxs))
    # use fromiter to convert to numpy as it is ~2x quicker than conversion
    # from list or tuple
    return np.fromiter(shift, dtype=dtype), errors


def estimate_shifts(
    curr_regions: np.ndarray, next_regions: np.ndarray,
    ptrn_size: Tuple[int, ...], dtype: type = np.int16
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the shift between spacially aligned regions in curr_regions to 
    next_regions. For each shift, there is also a 1D array of errors returned. These
    errors correspond to all possible shifts in the registration process.

    Args:
        curr_regions: a D+1 array with shape (N, R_1, ..., R_D) where N is the 
            number of regions and (R_1, ..., R_D) is the shape of a single 
            region. The search patterns are extracted from these regions
        next_regions: a D+1 array with the same shape are curr_regions. The 
            patterns extracted from curr_regions are searched for in these 
            regions. The regions are paired up in tandem i.e. pattern extracted
            from region curr_regions[0,...] will be searched for in 
            next_regions[0,...]
        ptrn_size: tuple of length D defining the size of the pattern to 
            extract and search for
        dtype: the data-type of the returned array

    Returns:
        an (N, D) array giving the estimated shift between pairs of regions 
            in each direction
        an (N, M) array of errors - for each shift, we have an array of errors
            representing the possilbe shifts in the registration process 
    """

    # ensure both region arrays have the same shape, otherwise cannot do a
    # 1-to-1 comparison
    if curr_regions.shape != next_regions.shape:
        raise ValueError(
            f'curr_regions and next_regions must have the same shape. Instead got: {curr_regions.shape} vs {next_regions.shape}'
        )
    # ensure ptrn_size specifies a size for each dimension in a single region
    if len(ptrn_size) != (curr_regions.ndim - 1):
        raise ValueError(
            f'ptrn_size must specify for each dimension in region. Expected: {curr_regions.ndim-1}, got: {len(ptrn_size)}'
        )

    # iterate over all curr, next region pairs and estimate the spacial 
    # shift of the extracted pattern
    shifts_and_errors = [_estimate_shift(curr_region, next_region, ptrn_size, dtype)
              for curr_region, next_region in zip(curr_regions, next_regions)]
    shifts_list = [i[0] for i in shifts_and_errors]
    errors_list = [i[1] for i in shifts_and_errors]
    # stack shifts into a single numpy array
    shifts = np.stack(shifts_list, axis=0)
    errors = np.stack(errors_list, axis=0)
    
    return shifts, errors


def _process_shifts(
    inputs_tuple: Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple wrapper around the estimate_shifts functon purely to allow the 
    inclusion of a progress bar during processing.

    Args:
        The inputs tuple contains the args of estimate_shifts:
            curr_regions
            next_regions
            pattern_size

    Returns:
        The outputs are the same as estimate_shifts:
            shifts
            errors
    """
    return estimate_shifts(*inputs_tuple)
    

def extract_search_regions(
    frames: np.ndarray, max_window: int, ptrn_size: Tuple[int, ...], region_size: Tuple[int, ...]
) -> np.ndarray:
    """ Extract the search regions from the frames array. 
    
    Args:
        frames: sequence of frames, array of shape (N_t, R_1, ..., R_D), where N_t is the number of
        frames and (R_1, ..., R_D) are the sizes of frame in each dimension.
        max_window: the number of frames over which to take a max value
        ptrn_size: tuple of length D defining the size of the pattern to extract and search for 
        region_size: tuple of length D definig the are over which to search for the pattern

    Returns:
        regions: (2+D)-dim array where for D=2, the shape is (aggregate frame, region, height, width)
    """
    # take every max_window frames in the time dimension and apply a maxpool operation 
    # over the time dimension. This is done to increase the signal in each aggregate 
    # frame to make the registration easier. The signal must be  increased in this way 
    # to account for beads that may submerge between frames and re-emerge. 
    regions = patch.extract(
        frames, (max_window, *frames.shape[1:])
    ).max(axis=1)
    # for each aggregate view extract search regions of size region_size with step 
    # size ptrn_size
    regions = patch.extract(
        regions, (regions.shape[0], *region_size), 
        (regions.shape[0], *ptrn_size), mode='valid'
    )
    # move aggregate frame view to first dimension so we have an array 
    # (aggregate frame, region, region-height, region-width)
    regions = np.moveaxis(regions, 1, 0)  
    
    return regions


def process_sequential_frames(
    regions: np.ndarray, ptrn_size: Tuple[int, ...], ncpus: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the shifts between sequential pairs of regions in parallel, using the 
    estimate_shifts function

    Args:
        regions: array of regions of shape (N_t, N_regions, region-height, region-width), 
        assuming 2-D frames
        ptrn_size: tuple of length 2 defining the size of the pattern to extract and search for 
        ncpus: number of cpus to parallelise over

    Returns:
        frame_shifts: array of shape (N_t-1, n_regions, 2*) giving 2*-D shift
        frame_errors: array of shape (N_t, n_regions, K) where K is the number of 
        possible matches for the central pattern in each sequential region.
    """
    input_list = zip(
        regions[:-1], regions[1:], [ptrn_size for _ in range(regions.shape[0]-1)]
    )
    with Pool(ncpus) as p:
        frame_shifts_and_errors = list(tqdm(p.imap(_process_shifts, input_list)))
    # split output into shifts and errors associated with the shifts
    frame_shifts = np.stack([i[0] for i in frame_shifts_and_errors], axis=0)
    frame_errors = np.stack([i[1] for i in frame_shifts_and_errors], axis=0)
    
    return frame_shifts, frame_errors


def run_registration_process(
    frames: np.ndarray, max_window: int, ptrn_size: Tuple[int, ...], region_size: Tuple[int, ...],
    ncpus: int
) -> Dict:
    """Run the registration process on a sequence of frames and return the following in a dict
    
    Args:
        frames: sequence of frames, array of shape (N_t, R_1, ..., R_D), where N_t is the number of
        frames and (R_1, ..., R_D) are the sizes of frame in each dimension.
        max_window: the number of frames over which to take a max value
        ptrn_size: tuple of length D defining the size of the pattern to extract and search for 
        region_size: tuple of length D definig the are over which to search for the pattern
        ncpus: number of cpus to parrallelise the estimate_shifts function

    Returns:
        a dictionary containing the following:
            - shifts between sequential (given max_window) frames
            - errors
            - unpad slice to remove the border pixels created in the patching process
            - sub shape of the downsampled shifts
    """

    # Extract the search regions
    regions = extract_search_regions(frames, max_window, ptrn_size, region_size)

    # process each sequential pair of frames in parallel
    # NOTE: each pair of frames takes ~3GB of RAM and ~1 minute per CPU to process
    frame_shifts, frame_errors = process_sequential_frames(regions, ptrn_size, ncpus)
    
    # the registration process effectively downsamples the orignal 
    # frames by applying a pooling operation over ptrn_size blocks, 
    # going from (5, 5, 1) -> (1, 1, 2) 
    # we therefore need to calculate the size of the downsampled frames
    # this assumes the original frames and the ptrn_size are both square
    sub_shape = int(np.sqrt(frame_shifts.shape[1]))

    # construct a slice to remove the border pixels lost during the 
    # patching process used to calculate the frame shifts
    # this slice will be used to remove border pixels in other downsampled
    # arrays such as the motion mask or sub_max_frame 
    unpad = (frames.shape[1] - (sub_shape * ptrn_size[0])) // 2
    unpad_slice = slice(unpad, -unpad)
    
    # reshape the calculated frame shifts into downsampled images
    # we get 1 frame shift per sequential pair of aggregated frames 
    # with shape (height, width, 2) the last dimension the is
    # dimensionality of the calculated velocity vector for the 
    # given pattern location
    shifts = frame_shifts.reshape(
        frame_shifts.shape[0], sub_shape, sub_shape, 2
    )
    errors = frame_errors.reshape(
        frame_errors.shape[0], sub_shape, sub_shape, (region_size[0]-ptrn_size[0]+1)**2
    )

    return {
        'shifts': shifts,
        'errors': errors,
        'unpad_slice': unpad_slice,
        'sub_shape': sub_shape
    }


def calculate_min_max_speed(
    ptrn_size: Tuple[int, ...], region_size: Tuple[int, ...]
) -> Tuple[float, float]:
    """Given a reigon and pattern size, return the min and max possible speeds that 
    can be calcualted in the registration process. Function will assume D=2

    Args:
        ptrn_size: tuple of length D defining the size of the pattern to extract and search for 
        region_size: tuple of length D definig the are over which to search for the pattern

    Returns:
        minimum speed 
        maximum speed
    """
    min_mag = 0 # min value is always 0 (i.e. no movement)

    # the max possible value is dependent on the region and pattern size used during the registration process. 
    # i.e. we can only detect movement  upto the size of the gap between the pattern boundary and the search 
    # region boundary  
    max_mag = [
        (r_size - p_size)/2 for r_size, p_size in zip(ptrn_size, region_size)
    ]
    max_mag = np.sqrt(sum(m_size**2 for m_size in max_mag))

    return min_mag, max_mag


def calculate_mean_velocity_field(shifts: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate a normalised average velocity vector field and speed scalar field given
    an array of shifts.

    Args:
        Array of shifts, of shape (N_t, X, Y, 2), assuming the shifts array is for a 2D field.

    Returns:
        Dictionary containing:
            normalised mean velocity vector field of shape (X, Y, 2)
            mean speed scalar field of shape (X, Y)
    """
    # calculate the mean velocity vector over time
    mean_shifts = shifts.mean(axis=0)
    # calculate the magnitude of each velocity vector
    mags = np.linalg.norm(mean_shifts, axis=-1)

    # construct a normalised vector field
    # need to replace magnitude of 0 length vectors with inf so 
    #  normalization gives sensible results
    mags = np.where(mags == 0, np.inf, mags)
    norm_shifts = mean_shifts/np.stack([mags]*2, axis=-1)
    # revert replacement of inf back to 0 so the subsequent colour 
    #  mappings are sensible
    mags = np.where(mags == np.inf, 0, mags)

    return {
        'normalised_velocity': norm_shifts,
        'speed': mags
    }


def run_registration_process_for_auxillary_analysis(
    frames: np.ndarray, movement_mask: np.ndarray, max_window: int,
    ptrn_size: Tuple[int, ...], region_size: Tuple[int, ...], ncpus: int
) -> Dict[str, np.ndarray]:
    """Wrapper function to run both the run_registration_process function and calculate_mean_velocity_field
    for the auxillary analysis where errors are considered in more detail.

    Args:
        frames: sequence of frames, array of shape (N_t, R_1, ..., R_D), where N_t is the number of
        frames and (R_1, ..., R_D) are the sizes of frame in each dimension. Assumption D=2
        movement_mask: boolean array of shape (R_1, ..., R_D) showing regions of motion
        max_window: the number of frames over which to take a max value
        ptrn_size: tuple of length D defining the size of the pattern to extract and search for 
        region_size: tuple of length D definig the are over which to search for the pattern
        ncpus: number of cpus to parrallelise the estimate_shifts function

    Returns:
        a dictionary containing the following:
            - vx: normalised velocity in x direction
            - vy: normalised velocity in y direction
            - mags: speed scalar field
            - sub_mask: downsampled motion mask
            - errors
    """
    reg_process = run_registration_process(
        frames, max_window, ptrn_size, region_size, ncpus
    )
    # deconstruct the ouput dictionary
    shifts = reg_process['shifts']
    errors = reg_process['errors']
    unpad_slice = reg_process['unpad_slice']
    sub_shape = reg_process['sub_shape']
    
    # construct the downsampled movement mask
    sub_mask = patch.extract(movement_mask[unpad_slice, unpad_slice], ptrn_size).max(axis=(-1, -2))
    sub_mask = sub_mask.reshape(sub_shape, sub_shape).astype(bool)

    # calculate average velocity field from shifts array
    velocity_fields = calculate_mean_velocity_field(shifts)
    norm_shifts = velocity_fields['normalised_velocity']
    mags = velocity_fields['speed']
    
    return {
        'vx': norm_shifts[:,:,1],
        'vy': norm_shifts[:,:,0],
        'mags': mags,
        'sub_mask': sub_mask,
        'errors': errors
    }


def calculate_angles_for_validation(
    forwards_velocity: np.ndarray, backwards_velocity: np.ndarray
) -> np.ndarray:
    """Produce an array of angles showing the angles between the positive_velocity vectors
    and the negative of the backwards velocity vectors.
    
    Args:
        forwards_velocity: array of normalised velocities in the forwards time direction.
        For 2D velocities the shape of the array is (height, width, 2)
        backwards_velocity: array of normalised velocities in the backwards time direction.
        For 2D velocities the shape of the array is (height, width, 2)
    
    Returns:
        Array of angles of shape (height, width) in degrees
    """
    # consider regions in either velocity vector field where both the vx and vy values are 
    # zero these are regions of noise as detected in the registration process and when 
    # calcualting the cosine-difference of the angles between forwards and backwards 
    # pass vectors. these regions of 0-vectors lead to 
    # cosine(diff) = 0 => diff = 90 so we get a large number of 90 degree angle errors
    # we must be careful in considering these zero vectors.
    # we can create a boolean mask to highlight the pixels where there is a zero vector 
    # in BOTH forwards AND backwards pass this boolean mask will be used to hard set 
    # the angle error to zero
    zero_mask = (
        ((backwards_velocity[...,0]==0) & (backwards_velocity[...,1]==0)) & 
        ((forwards_velocity[...,0]==0) & (forwards_velocity[...,1]==0))
    )
    # get differnce_cosine - dot product of noramlised vel vectors
    cosines = (
        (forwards_velocity[...,0]*(-1)*backwards_velocity[...,0]) + 
        (forwards_velocity[...,1]*(-1)*backwards_velocity[...,1]) 
    ) 
    # get rid of numerical errors so that |cosines| < 1
    cosines[cosines>1] = 1
    cosines[cosines<-1] = -1
    # apply zero mask, calculate arcosine, convert angles to degrees
    cosines[zero_mask] = 1
    angles = (180./math.pi) * (np.arccos(cosines))
    
    return angles
