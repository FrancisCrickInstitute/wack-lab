import numpy as np
class path:
    def __init__(self, label):
        self.live_path = True
        self.P_pred = []
        self.P_est = []
        self.x_pred = []
        self.x_est = []
        self.z_meas = []
        self.z_meas_latest = (np.empty(shape=(0, 2)),[]) # Plase to store the latest measurement as (z_meas, frame_idx)
        self.label = label
        self.propagation_counter = 0 # Number of time steps the track has been propagated

    def __repr__(self):
        x_est_last, _, frame_idx_est = self.return_last_estimates()
        if not isinstance(x_est_last, list):
            x_est_last = x_est_last.ravel()
        _, _, frame_idx_pred = self.return_last_predictions()    

        disp_string = (f"label: {self.label}, alive: {self.live_path},frame_idx of last pred.: {frame_idx_pred}, frame_idx of last est.: {frame_idx_est}\n "
        f"Last state estimate: {x_est_last}")
        return disp_string   

    def add_predictions_to_track(self, x_pred, P_pred, frame_idx):
        self.P_pred.append((P_pred, frame_idx))
        self.x_pred.append((x_pred, frame_idx))

    def add_estimates_to_track(self, x_est, P_est, frame_idx):
        self.P_est.append((P_est, frame_idx))
        self.x_est.append((x_est, frame_idx))
    
    def return_last_state_estimate(self):
        if self.x_est:
            x_est_last, frame_idx = self.x_est[-1]
        else:
            frame_idx= -1
            x_est_last = self.x_est
        return (x_est_last, frame_idx)
        

    def return_last_covariance_estimate(self):
        if self.P_est:
            P_est_last, frame_idx = self.P_est[-1]
        else:
            frame_idx= -1
            P_est_last = self.P_est
        return (P_est_last, frame_idx)

    def return_last_estimates(self):
        (P_est, frame_idx_P) = self.return_last_covariance_estimate()
        (x_est, frame_idx_x) = self.return_last_state_estimate()

        # Check that they are from the same frame index
        if frame_idx_x != frame_idx_P:
            raise ValueError("The latest state and covariance estimate are not from the same frame index.")
        return (x_est, P_est, frame_idx_x)

    def return_last_state_prediction(self):
        if self.x_pred:
            x_pred_last, frame_idx = self.x_pred[-1]
        else:
            frame_idx= -1
            x_pred_last = self.x_pred
        return (x_pred_last, frame_idx)
        

    def return_last_covariance_prediction(self):
        if self.P_pred:
            P_pred_last, frame_idx = self.P_pred[-1]
        else:
            frame_idx= -1
            P_pred_last = self.P_pred
        return (P_pred_last, frame_idx)

    def return_last_predictions(self):
        (P_pred, frame_idx_P) = self.return_last_covariance_prediction()
        (x_pred, frame_idx_x) = self.return_last_state_prediction()

        # Check that they are from the same frame index
        if frame_idx_x != frame_idx_P:
            raise ValueError("The latest state and covariance prediction are not from the same frame index.")
        return (x_pred, P_pred, frame_idx_x)

    def get_estimate_by_frame_idx(self, idx_frame):
        estimate_for_frame = [est for est in self.x_est if est[1]==idx_frame]
        if estimate_for_frame:
            output = estimate_for_frame[0]
        else:
            output = (estimate_for_frame,-1)
        return output

    def reset_measurement(self):
        self.z_meas_latest = (np.empty(shape=(0, 2)),[])


def get_new_unique_label(paths):
    # Get a label for a new paths that is distinct from the old paths
    labels = []
    if not isinstance(paths, list):
        paths = [paths]
    
    if paths:
        for path in paths:
            labels.append(path.label)


        new_label = max(labels) + 1
    else:
        new_label = 0

    return new_label

def get_path_by_label(paths, label):
    if not isinstance(paths, list):
        paths = [paths]
    
    if paths:
        path_obj = [path_obj for path_obj in paths if path_obj.label == label]
    else:
        raise ValueError("Can't get label from empty paths.")

    if not path_obj:
        raise ValueError(f"No path available with label {label}")  
    
    return path_obj[0]

def get_live_paths(paths):
    live_paths = paths.copy()
    return([path_obj for path_obj in live_paths if path_obj.live_path])

def get_path_labels(paths):
    return([path_obj.label for path_obj in paths])     

def check_paths_alive(paths, prop_thd):
    updated_paths = []
    
    # Loop over all paths and propagate the estimate
    for path_obj in paths:    
        if path_obj.propagation_counter > prop_thd:
            path_obj.live_path = False
        updated_paths.append(path_obj)
    return updated_paths       