import os
from typing import List
from collections import defaultdict

import cv2
import imageio
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from rootpath import detect
from skimage import io
from tqdm import tqdm
import re
import json
from skimage.morphology import disk, closing
from joblib import Parallel, delayed


def here(relative: bool = False) -> str:
    """Find the root directory of the project

    Thin wrapper around the detect function to always use the ".here" file to 
    identify the project root directory, and always starts its search from the 
    folder of the executing file (i.e. __file__)

    Args:
        relative: if True will return the path relative to the current working
            directory (default False)

    Returns:
        a str of the path to the root directory of this project

    """

    # get absolute path to project root
    path = detect(current_path=__file__, pattern='.here')
    if relative:
        # construct the path to the project directory relative to the current
        # working directory
        path = os.path.relpath(path, os.getcwd())

    return path


def imshow(*args, figsize: int = None, **kwargs):
    if figsize is not None:
        plt.figure(figsize=(figsize, figsize))
    plt.imshow(*args, **kwargs)


def _frames_to_video(
    filename: str, frames: np.ndarray, codec: str, fps: float = None,
    duration: int = None
) -> None:
    """Save a collection frames as a video file

    Args:
        filename: name (including path) of the output video file
        frames: the collection of frames which make up the video. The array can
            be (F, H, W) corresponds to a greyscale video 
            where F is the number of frames and (H, W) are the height and width
            of the video. When it is (F, H, W, 3) it should correspond to a 
            colour video in BGR format. When it is (F, H, W, 4) it should 
            correspond to a colour video in BGRA format
        codec: 4-charcter code of the codec used to compress the frames
        fps: frame rate of the video. One of fps and duration must be provided
        duration: length of the video in seconds. One of fps and duration must 
            be provided. If fps is given duration is ignored

    """
    if fps is None:
        if duration is None:
            raise TypeError('One of fps or duration must be provided')
        fps = frames.shape[0] / duration

    # extract the dimensions of the video. Need to reverse H, W as OpenCV
    # expects dimensions to be in reverse order from numpy ordering
    frame_shape = frames.shape[1:3][::-1]

    # convert grayscale frames to gray colour in BGR space
    if frames.ndim == 3:
        frames = np.stack([frames]*3, axis=-1)

    try:
        # setup video codec and writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(filename, fourcc, fps, frame_shape)

        # iterate over first dimension of the array
        for frame in tqdm(frames):
            out.write(frame)
    finally:
        # wrap video creation operations in try/finally to ensure resource
        # release steps are always executed
        out.release()
        cv2.destroyAllWindows()


def frames_to_video(
    filename: str, frames: np.ndarray, fps: float = None,
    duration: int = None
) -> None:
    """Save a collection frames as a video file

    This functions supports 2 file formats:
        - MP4 file format, which uses a codec with lossy compression but
            results in a much smaller video file
        - AVI file format, which uses a codec with lossless compression but
            results in a much larger video file
    A rough comparison,  MP4 ~10MB <--> AVI ~2GB

    Args:
        filename: name (including path) of the file to write the video to
        frames: collection of frames which make up the video. The array 
            must have the dtype uint8. The array can be (F, H, W) 
            corresponds to a greyscale video where F is the number
            of frames and (H, W) are the height and width of the video. When it
            is (F, H, W, 3) it should correspond to a colour video in BGR
            format. When it is (F, H, W, 4) it should correspond to a colour
            video in BGRA format.
        fps: frame rate of the video. One of `fps` and `duration` must be 
            provided
        duration: length of the video in seconds. One of `fps` and `duration` 
            must be provided. If fps is given duration is ignored

    Raises:
        ValueError: extension in `filename` is not of the supported types 
    """
    codec = None
    if filename.lower().endswith('.mp4'):
        codec = 'avc1'
    if filename.lower().endswith('.avi'):
        codec = 'ffv1'

    if codec is None:
        raise ValueError(
            f'Only mp4 and avi file formats supported, instead got: {filename}'
        )
    _frames_to_video(filename, frames, codec=codec, fps=fps, duration=duration)


def frames_to_stack(filename: str, frames: np.ndarray) -> None:
    """Save a collection of frames as a stack of images

    Args:
        filename: location and name of the file to write the images to
        frames: numpy array of shape: (N, H, W), (N, H, W, 3), or (N, H, W, 4)
            where N is the number of frames, and H,W are the dimensions of the 
            frames
    """
    ext = os.path.splitext(filename)[-1]
    if ext.lower() not in {'.tif', '.tiff'}:
        raise ValueError('Only TIFF file format supported')

    imageio.mimwrite(filename, frames)


def frames_from_stack(image_stack_file: str) -> np.ndarray:
    """Extracts the frames of a stacked image file as a numpy array

    Args:
        image_stack_file: path to the image stack file to extract the frames 
            from

    Returns:
        stacked numpy array, where the first dimension is the frame count.
    """

    frames = io.imread(image_stack_file)

    return frames


def frames_from_video(video_file: str) -> np.ndarray:
    """Extracts the frames from a video file

    Args:
        video_file: path to the image stack file to extract the frames from

    Returns:
        stacked numpy array, where the first dimension is the frame count.
    """

    frames = []

    try:
        # start video capture
        cap = cv2.VideoCapture(video_file)

        # load all frames from the video
        while True:
            # each call to read() returns a bool and an array
            # the bool indicates whether the end of the video has been reached
            # i.e. ret == False means no frame returned
            ret, frame = cap.read()

            # if the no frame has been returned break out of the loop
            # need an explicit break as additional calls the read() will just start
            # looping over the video
            if not ret:
                break

            frames.append(frame)

    finally:
        # want to ensure capture resources are always released
        # so they are called in a finally block
        cap.release()
        cv2.destroyAllWindows()

    # stack the frames into a single numpy array and return them
    return np.stack(frames, axis=0)


def get_folder_information(exp_id: str, hbec_root: str, neighbourhood_size: int=9):
    """Find file id's for a given experiment ID and hbec root folder.
    This function returns the identifiers of the individual samples within an experiments,
    as well as the directory where the registration results and spatial autocorrelation results are stored.
    Args:
        exp_id: string containing the experiment identifier
        hbec_root: string containing the root of the hbec data
        neighbourhood_size: int defining the neighbourhood size for the Geary's C index calculation. Default is set to 9.
    Returns:
        file_ids: Tuple containing experiment ID, group ID and batch ID
        register_root_path: string containing the root of the registration data
        sac_root_path: string containing the root of the spatial autocorrelation data
    """
    # declare the various output directories
    processed_root = os.path.join(hbec_root, exp_id)
    register_root = os.path.join(processed_root, 'register')
    sac_root = os.path.join(processed_root, 'spatial_auto_correlation', f'neighbor_size_{neighbourhood_size}_geary')
    REGEX = r'([a-zA-Z_0-9]*?)_([0-9]{1,}_[0-9]{1,})_.*'
    pattern = re.compile(REGEX)

    # find all relevant data files in the data directory 
    dir_names = sorted([_d for _d in os.listdir(sac_root) if os.path.isdir(os.path.join(sac_root, _d))])

    # identify the group ID and batch ID for each file
    file_ids = [(_f, pattern.match(_f)) for _f in dir_names]
    file_ids = [(match[0], *match[1].groups()) if match[1] is not None else match for match in file_ids]
    return register_root, sac_root, file_ids


def points_to_markers(points: dict, shape: tuple) -> np.ndarray:
    """Convert dictionary with centers of mass to mask.

    Args:
        points: dictionary with centers of mass of segments.
    Returns:
        Mask with centers of mass of segments.
    """
    markers = np.zeros((shape[0], shape[1]))
    for label, value in points.items():
        markers[value[0], value[1]] = label
    return markers


def markers_to_pts(markers: np.ndarray) -> dict:
    """Convert mask to dictionary with centers of mass

    Args:
        markers: Mask with centers of mass of segments.
    Returns:
        Dictionary with centers of mass of segments.
    """
    pts = dict({})
    for j in range(1, int(np.max(markers))+1):
        x, y = sp.ndimage.center_of_mass(markers == j)
        pts.update({j: [x, y]})
    return pts


def save_video(raw_frames: np.ndarray, paths: np.ndarray, video_id: str):
    """Plot paths over frames and save as video

    Args:
        raw_frames: collection of frames which make up the video. The array
            must have the dtype uint8. The array can be (F, H, W) 
            corresponds to a greyscale video where F is the number
            of frames and (H, W) are the height and width of the video. When it
            is (F, H, W, 3) it should correspond to a colour video in BGR
            format. When it is (F, H, W, 4) it should correspond to a colour
            video in BGRA format.
        paths: stacked numpy array, where the first dimension is the frame count.
        video_id: name (including path) of the file to write the video to
            without extension.
    """
    
    # initialise empty defaultdict
    by_frame_tracked_pts = defaultdict(list)

    # store lists of sequential points in each path grouped
    # by their frame index
    for idx_frame in range(raw_frames.shape[0]):
        for path_obj in paths:
            current_estimate,_ = path_obj.get_estimate_by_frame_idx(idx_frame)
            next_estimate, _ = path_obj.get_estimate_by_frame_idx(idx_frame+1)

            if isinstance(current_estimate, np.ndarray) and isinstance(next_estimate, np.ndarray):
                prev_pt = np.round(current_estimate.ravel()[:2]).astype(int)[::-1]
                curr_pt = np.round(next_estimate.ravel()[:2]).astype(int)[::-1]
                by_frame_tracked_pts[idx_frame].append([tuple(prev_pt), tuple(curr_pt)])
                
    mask = np.zeros(raw_frames.shape[1:], dtype=np.uint8)
    
    # go through all frames
    for frame_idx in tqdm(range(raw_frames.shape[0])):

        frame_paths = by_frame_tracked_pts[frame_idx]

        if frame_paths:
            for prev_pos, curr_pos in frame_paths:
                
                # draw a pink line where the points have been
                mask = cv2.line(mask, curr_pos, prev_pos, (255, 0, 255), 3)
                
                # draw blue circle where the current point is, 
                # with radius 10 and thickness -1
                raw_frames[frame_idx,...] = cv2.circle(raw_frames[frame_idx,...],
                                                       curr_pos,
                                                       10,
                                                       (255, 0, 0),
                                                       -1)
                # add path to current frame
                raw_frames[frame_idx,...] = cv2.add(raw_frames[frame_idx,...], mask)
        else:
            # add path to current frame
            raw_frames[frame_idx,...] = cv2.add(raw_frames[frame_idx,...], mask)
    
    # save frames to file
    frames_to_video(video_id + ".mp4", raw_frames, fps=19)
    

def get_masks(register_root, sac_root, name):
    """Given the paths to the data return the coordination mask, movement mask, speed mask and 
    roi mask. Note that the geary speed mask is not considered since coordination
    in direction of motion is more important than coordination in speed of motion.
    Also return the sub-movement mask to avoid loading it twice. 
    Args:
        register_root: path to registration data
        sac_root: path to spatial autocorrelation data
        name: name of the folder that represents a well
    Returns:
        numpy array - coordination mask
        numpy array - motion mask
        numpy array - mags matrix
        numpy array - roi mask
    """
    # given a specific file, set the path of this file
    file_root_path = os.path.join(register_root, name)

    # load the registration parameters for specific file
    with open(os.path.join(file_root_path, 'params.json'), 'r') as params_file:
        params = json.load(params_file)    

    # get shift vectors and normalise according to the max_window value
    shifts = np.load(os.path.join(register_root, name, 'shifts.npy'))
    shifts = shifts/params['max_window']

    # load the motion mask and roi mask (from the registration process)
    sub_movement_mask = np.load(os.path.join(register_root, name, 'mask.npy'))
    roi_mask = np.load(os.path.join(register_root, name, 'roi.npy'))

    # load the geary masks from the spatial autocorrelation directory
    geary_vx_mask = np.load(os.path.join(sac_root, name, 'vx_mask.npy'))
    geary_vy_mask = np.load(os.path.join(sac_root, name, 'vy_mask.npy'))

    # calculate the mean velocity vector over time
    mean_shifts = shifts.mean(axis=0)
    # calculate the magnitude of each velocity vector
    mags_matrix = np.linalg.norm(mean_shifts, axis=-1)

    min_mags_mask = (mags_matrix > np.average(mags_matrix[~sub_movement_mask & roi_mask]))    
    coordination_mask = (geary_vx_mask & geary_vy_mask) & sub_movement_mask & min_mags_mask
    coordination_mask = closing(coordination_mask.astype(np.int8), disk(1)).astype(bool)

    return coordination_mask, sub_movement_mask, mags_matrix, roi_mask


def generate_mask_and_speed_summary_df(experiment_id, neighbourhood_size: int=9):
    """Creates a summary df of mask proportions and speed measurements for a specific experiment
    Args:
        experiment_id: string of directory name for processed experiment data
    Returns:
       dataframe, with columns containing the experiment, group and batch ID, 
       the mean speed for that experiment, the type of mask that is applied and the mask ratio.
       zero_mask_df, with columns containing experiment, group, batch, mask-type - this helps
       the user in identifying wells where no speed is recorded since the mask is not detected
    """
    # get the paths for the registration data and spatial autocorrelation data
    hbec_root = os.path.join(here(), 'data', 'processed', 'hbec')
    register_root, sac_root, file_ids = get_folder_information(experiment_id, hbec_root, neighbourhood_size) 

    # loop through the file_ids and append the summary_df
    summary_df = []
    # store datapoints for which mask ratio is 0
    zero_mask_ratio_list = []
    for name, group_id, batch_id in file_ids:
        # produce the coordination mask and movement mask and mags_matrix
        coordination_mask, sub_movement_mask, mags_matrix, roi_mask = \
            get_masks(register_root, sac_root, name)

        # create dictionary of masks, to loop through and store mask proportions and speeds in df
        mask_dict = {
            '1_motion':sub_movement_mask,
            '2_coordination': coordination_mask,
        }   
        # loop through the masks
        for mask_type, mask in mask_dict.items():
            mags = mags_matrix[mask]
            if mask_type == '2_coordination':
                mask_ratio = mask.sum()/sub_movement_mask.sum()
            else:
                mask_ratio = mask.sum()/roi_mask.sum()
            
            # if mask_ratio is 0, save the datapoint details to the 
            # zero_mask_ratio_list list
            if mask_ratio == 0:
                zero_mask_ratio_list.append({
                    'experiment_name': experiment_id,
                    'group_id': group_id,
                    'batch_id': batch_id,
                    'mask_type': mask_type,
                })

            df = pd.DataFrame({
                'experiment_name': experiment_id,
                'experiment': get_experiment_short_name(experiment_id),
                'name': name,
                'group_id': group_id,
                'batch_id': batch_id,
                'speed': mags,
                'mask_type': mask_type,
                'mask_ratio': mask_ratio
            })
            summary_df.append(df)

    # rename knockout types
    df = pd.concat(summary_df)
    renamed_df = rename_knockout_types(df)

    return renamed_df, pd.DataFrame(zero_mask_ratio_list)


def get_experiment_short_name(experiment_id):
    """Given an experiment ID, shorten the ID so that the summary graphs
    can display them in the legend.
    Args:
        experiment_id: string of directory name for processed experiment data
    Returns:
       str, shortened experiment name.
    """
    if experiment_id == 'ELN14186_8_perc':
        experiment_nickname = '14186_8'
    elif experiment_id == 'ELN14186_6_perc':
        experiment_nickname = '14186_6'
    elif experiment_id == 'ELN16420-5_methylcellulose_1_5':
        experiment_nickname = '16420_methyl'
    elif experiment_id == 'ELN19575-4':
        experiment_nickname = '19575_4'
    elif experiment_id == 'ELN19575-5':
        experiment_nickname = '19575_5'
    else:
        experiment_nickname =  '_'.join(experiment_id.split('-')[-1].split('_')[0:2])

    return experiment_nickname


def generate_summary_df_for_all_experiments(experiment_list, ncpus):
    """Creates a summary df of mask proportions and speed measurements of all
    experiments in the experiment_list.
    Args:
        experiment_list: list of experiment directory names
        ncpus: parallelise over ncpus
    Returns:
       dataframe, with columns containing the experiment, group and batch ID, 
       the mean speed for that experiment, the type of mask that is applied and the mask ratio.
    """
    with Parallel(n_jobs=ncpus, verbose=20, backend='multiprocessing') as par:
        summary_and_zero_mask_df = par(
            delayed(generate_mask_and_speed_summary_df)(exp) 
            for exp in experiment_list
        )

    # summary_df = list(itertools.chain.from_iterable(summary_df))
    summary_df = pd.concat([i[0] for i in summary_and_zero_mask_df])
    zero_mask_ratio_df = pd.concat([i[1] for i in summary_and_zero_mask_df])

    return summary_df, zero_mask_ratio_df

rename_tuple_list = [
    ('NT_PBS', 'NT'),
    ('g1_PBS', 'g1'),
    ('g1_ON', 'g1'),
    ('NT_ON', 'NT'),
    ('^DNA$', 'DNA_gAA'),
    ('DNAI1', 'DNA_gAA'),
]
def rename_knockout_types(df, rename_tuple_list: list=rename_tuple_list):
    """Function to rename knockout types in summary_df
    Args:
        df: summary dataframe
        rename_tuple_list: list of tuples with the names to replace
    Returns:
        summary df
    """
    for i in rename_tuple_list:
        df.group_id=df.group_id.str.replace(i[0],i[1])
    
    return df
