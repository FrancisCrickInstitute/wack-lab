from typing import List, Tuple

import numpy as np
import scipy as sp

from . import track, consts
from . import paths as paths_fncs

def matmul_trans(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Simple wrapper to perform matrix operation ABA.T

    Args:
        A: 2D numpy array with shape (J, K)
        B: 2D numpy square array with shape (K, K)

    Returns:
        2D numpy array of shape (J, J)

    """
    res = np.matmul(A, B)
    res = np.matmul(res, A.T)
    return res


def initialise_point(
    measured: np.ndarray, *args: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialise the Kalman filter state for a new point

        Args:
            measured: array of shape (M,), containing measured values of the
                state
            *args: ordered list, of length N, of standard deviations for each
                aspect of the full state

        Returns:
            initialised Kalman filter state:
                - x_est: (S, 1) numpy array containing the estimated state
                - x_pred: (S, 1) numpy array containing the predicted state
                - P_pred: (S, S) numpy array containing the covariance matrix 
                    of the predicted state
                - P_est: (S, S) numpy array containing the covariance matrix 
                    of the estimated state
            where S = M * N

    """
    measured_size = measured.shape[0]
    state_size = measured_size * len(args)

    # initialise state prediction and estimate
    x_est, x_pred = np.zeros((state_size, 1)), np.zeros((state_size, 1))
    x_est[:measured_size, 0], x_pred[:measured_size, 0] = measured, measured

    # initialise covariance matrices for the estimated and predicted state
    P_pred = [np.identity(measured_size) * (uncert**2) for uncert in args]
    P_pred = sp.linalg.block_diag(*P_pred)
    P_est = P_pred.copy()

    return x_est, x_pred, P_pred, P_est

def align_measurements_to_paths(paths, curr_pts, H, R):
    live_paths_to_align = paths_fncs.get_live_paths(paths)
    # Preallocate matrix to calculate assignment from current points to paths
    cost_mat = np.empty(shape=(len(curr_pts),len(live_paths_to_align)))
    # drop dimensions of size 1
    curr_pts = np.squeeze(curr_pts)

    path_labels = []
    for path_idx, path_obj in enumerate(live_paths_to_align):
        x_pred, P_pred, frame_idx_pred = path_obj.return_last_predictions()
        path_labels.append(path_obj.label)
        for curr_pt_idx, pt in enumerate(curr_pts):
            u = pt.reshape(-1, 1)
            v = np.matmul(H, x_pred)
            VI = np.linalg.inv( matmul_trans(H, P_pred) + R)
            cost_mat[curr_pt_idx, path_idx] = sp.spatial.distance.mahalanobis(u, v, VI)
    
    # use Munkres algorithm to align the current points with the paths
    curr_pts_idxs, path_idxs = sp.optimize.linear_sum_assignment(cost_mat)

    path_labels = [path_obj.label for path_obj in live_paths_to_align]

    # Reorder labels according to alignment
    aligned_path_labels = [path_labels[idx] for idx in path_idxs]
    
    return (curr_pts_idxs, aligned_path_labels)

def assign_measurements_to_paths(paths, curr_pts, curr_pts_idxs, path_labels, idx_frame):
    
    # Update  the measurementes for the paths that match
    for idx_pts, label in zip(curr_pts_idxs, path_labels):

        # Find the current path in the list of paths
        path_obj = paths_fncs.get_path_by_label(paths, label)
        matching_index = paths.index(path_obj)

        # Assign the latest measurement to the path
        paths[matching_index].z_meas_latest = (curr_pts[idx_pts], idx_frame)

    # Return new points so new paths can be generated
    new_pts = np.delete(curr_pts, curr_pts_idxs, axis=0)

    # Note that no new measurements are added for missing points
    
    return (paths, new_pts)


def append_new_paths(new_pts, measure_uncert, frame_idx, paths):
    # Create new paths for the new points
    if new_pts.shape[0]>0:
        for pt in new_pts:
            # Get new path label
            new_label = paths_fncs.get_new_unique_label(paths)
            new_path = paths_fncs.path(new_label)
            # Initialise estimates and predictions
            x_est, x_pred, P_pred, P_est = initialise_point(np.stack(pt, axis=0), *measure_uncert)
            new_path.add_predictions_to_track(x_pred, P_pred, frame_idx)
            new_path.add_estimates_to_track(x_est, P_est, frame_idx)
            paths.append(new_path)

    return paths

def propagate_paths(paths, frame_idx, PHI, GAM, Q):
    updated_paths = []
    
    # Loop over all paths and propagate the estimate
    for path_obj in paths:
        # Take the estimates for the last point of the path
        x_est, P_est, frame_idx_prev = path_obj.return_last_estimates()
        
        # If the last estimate is not from the previous frame, propagate the last prediction,
        # as there should always be one available from the previous frame.
        if not (frame_idx_prev + 1 == frame_idx):
            x_est, P_est, _ = path_obj.return_last_predictions()
            path_obj.propagation_counter = path_obj.propagation_counter + 1
            
        # Propagate forwards
        x_pred_step = np.matmul(PHI, x_est)
        P_pred_step = matmul_trans(PHI, P_est) + matmul_trans(GAM, Q) #why matmul_trans(GAM, Q) and not just Q?
        
        # Collect results for all points
        path_obj.add_predictions_to_track(x_pred_step, P_pred_step, frame_idx)
        updated_paths.append(path_obj)
    return updated_paths

def update_estimates_on_paths_with_measurements(paths, H, R, PHI):
# Update estimates for paths where a measurement is available:

    updated_paths = []
    for path_obj in paths:
        z_meas, idx_frame = path_obj.z_meas_latest
        if z_meas.shape[0]>0:
            
            # Update estimate based on whether a new measurement has been added and then clear that measurement.
            # Note that paths that have been newly initialised in this frame have an empty z_meas_latest and therefore will not be updated.
            # State Estimate Update (last box in table 4.2-1 in A.Gelb, Applied Optimal Estimation)
            x_pred_step, P_pred_step, _ = path_obj.return_last_predictions()
            
            # calculate the Kalman Gain
            innovation_cov = matmul_trans(H, P_pred_step) + R
            K = np.matmul(np.matmul(P_pred_step, H.T), np.linalg.inv(innovation_cov))

            # update the state estimate based on the measurement (valid_curr_pt)
            innovation = z_meas.reshape(-1, 1) - np.matmul(H, x_pred_step)
            x_est_step = x_pred_step + np.matmul(K, innovation)

            # update the covariance estimate based on the Gain
            P_est_step = np.identity(PHI.shape[0]) - np.matmul(K, H)
            P_est_step = np.matmul(P_est_step, P_pred_step)
            
            path_obj.add_estimates_to_track(x_est_step, P_est_step, idx_frame)
            
            # Clear the "used" measurement, store it in the history
            path_obj.z_meas.append((z_meas, idx_frame))
            path_obj.reset_measurement()
        
        updated_paths.append(path_obj)
            
    return updated_paths    