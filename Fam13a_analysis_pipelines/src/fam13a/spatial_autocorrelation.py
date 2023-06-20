import numpy as np
from skimage import morphology as sk_morph
from typing import Dict

from .image import patch


def create_neighbor_patch(patch_size: int) -> np.ndarray:
    """Create a square matrix of booleans which define the positions 
    of the neighboring elements for the central element. Use skimage morph.

    Args:
        patch_size: an integer defining the shape of the output array. Note that
        the output array will be a square matrix. The patch_size must be an odd integer.

    Returns:
        a (patch_size, patch_size) boolean array. The True locations define the neighbors of the 
        central element. The central element must be False by definition, since it 
        cannot be the neighbor of itself.

    Example:
        The array below shows the central element having neighbors above,
        below, left and right. 
        
        np.array([
            [0,1,0],
            [1,0,1],
            [0,1,0]
        ]).astype(bool)
    """

    # check if patch_size is odd and greater than 0
    if patch_size <= 0 or patch_size%2 == 0:
        raise ValueError('Patch size must be odd and greater than 0.')

    disk_radius = patch_size//2
    neighbor_patch = sk_morph.disk(disk_radius).astype(bool)

    # need to set the central element to False since it cannot be it's own neighbor
    # the central element has index value (disk_radius,disk_radius)
    neighbor_patch[disk_radius, disk_radius] = False

    return neighbor_patch


def _create_matrix_of_pseudo_indices(image_size: int) -> np.ndarray:
    """Create matrix of elements, where each element is the pseudo position in a flattened
    matrix (pseuod, because 1 is added to each element).
    
    Args:
        image_size: an integer defining the shape of the final matrix: (image_size, image_size)

    Returns:
        a 2D array of indices

    Example:
        For image_size=2, the result would be:

        np.array([
            [1,2],
            [3,4]
        ])
    """
    N = image_size*image_size
    flat_matrix_indices = np.arange(1, N+1)
    matrix_of_indices = flat_matrix_indices.reshape(image_size, image_size)

    return matrix_of_indices


def _populate_weights_neighbors_dict_given_patches(patches: np.ndarray, flat_neighbor_patch: np.ndarray):
    """For each patch in patches (given a neighbor-patch), determine the neighbors of the central
    element and populate a neighbors-dictionary. Also populate a weights dictionary (with 1.0 
    for each neighbor).

    Args:
        patches: (N, d1, d2) array of patches - resulting from image.patch.extract with padding of 
        0.0 values. N is the number of patches and d1,d2 are the dimensions of a single patch.
        flat_neighbor_patch: 1-D boolean array of length d1*d2 defining neighbors of central element
        in a patch.

    Returns:
        1. patches array where each patch has been replaced with an array defining it's position in the array
        2. neighbors dictionary
        3. weights dictionary
    """
    # initialise neighbors and weights dictionaries
    neighbors_dict = {}
    weights_dict = {}

    patch_size = patches[0].shape

    # for each patch, record the neighbors of the central index by 
    #  - using the neighbor patch to exclude non-neighbors from the patch
    #  - then exlude zero values (these are padded elements)
    #  - then subtract 1 from the patch elements
    # this for loop will add padded elements to the dictionaries - will need be removed later
    return_patches = patches.copy()
    for idx, patch_ in enumerate(patches):
        # exclusion of non-neighbors
        neighbors_only = patch_.flatten()[flat_neighbor_patch]
        # exclude zero values
        neighbors_only = neighbors_only[neighbors_only!=0]
        # subtract 1 to get true matrix indices
        neighbors_only -= 1
        # append neighbors and weights dict
        neighbors_dict[idx] = list(neighbors_only)
        weights_dict[idx] = list(np.ones_like(neighbors_only))
        # need to replace patches[idx] with a (patch_size,patch_size) zero-matrix
        # where the elements of the matrix are the id of the patch
        # this is for merging later (opposite of patch.extract)
        return_patches[idx] = np.full(patch_size, idx)

    return return_patches, neighbors_dict, weights_dict


def _remove_padded_elements_from_weights_neigbors_dicts(
    neighbors_dict: Dict[int, list], weights_dict: Dict[int, list], flat_merged_matrix: np.ndarray
    ):
    """Remove padded elements from the weights and neighbors dictionary. The elements in merged_matrix
    refer to patches that contain at least one non-padded element. Their elements can be used to filter
    the weights and neighbors dictionary. The keys in the dictionaries will also be reset.
    
    Args:
        neighbors_dict: dictionary containing nieghbors
        weights_dict: dictionary containing weights
        merged_matrix: result of image.patch.merge to remove padded elements

    Returns:
        re-indexed neighbors and weights dictionraies
    """

    neighbors_dict = {
        i:neighbors_dict[k] for i,k in enumerate(flat_merged_matrix)
    }
    weights_dict = {
        i:weights_dict[k] for i,k in enumerate(flat_merged_matrix)
    }

    return neighbors_dict, weights_dict


def create_spatial_dicts(image_size: int, neighbor_patch: np.ndarray) -> Dict:
    """Given an image_size and neigbor patch matrix, return two dictionaries:
        - neighbors dictionary: for each element in a (image_size, image_size) image matrix,
        a list of neighbors is defined via the indices
        - weights dictionary: for each neighbor defined in the neighbours dictionary,
        a weight is assigned (currently all 1.0). TODO: can create a create_weights_patch, where
        the weights are dependent on the euclidean distance to the centre. 
    
    Args:
        image_size: an integer defining the shape of the image for which spatial autocorrelation
        is to be calculated. 
        neighbor_patch: a square matrix defining the neigbors of a element

    Returns:
        a dictionary containing two dictionaries:
            - neighbors: for each element in a (image_size, image_size) image matrix,
            a list of neighbors is defined via the indices
            - weights: for each neighbor defined in the neighbours dictionary,
            a weight is assigned (currently all 1.0).
    """

    # first flattern the neigbors_patch
    flat_neighbor_patch = neighbor_patch.flatten()

    # create matrix of matrix indices 
    # 1 is added to the indices to distinguish the indices from the 
    # 0's added in the patch.extract function for padding
    matrix_of_indices = _create_matrix_of_pseudo_indices(image_size)

    # create patches of the matrix_of_indices 
    # the size of the patches will be the same as the size of the neighbor_patch
    # a constant mode will introduce padding of 0 elements to the patches - these need
    # to be distinguishable from the matrix-indices (hence why 1 is added above)
    patch_size = neighbor_patch.shape
    patches = patch.extract(matrix_of_indices, patch_size, [1,1], mode='constant')

    # get the weights and neighbors for the central element in each patch
    patches, neighbors_dict, weights_dict = _populate_weights_neighbors_dict_given_patches(
        patches, flat_neighbor_patch
    )

    # create merged matrix of patches, where each element in the merged_matrix matches
    # the useful keys (non-padded) in the neighbors and weights dictionaries
    # therefore this line removes the padded elements.
    merged_matrix = patch.merge(patches, (image_size,image_size), [1,1], padded = True)
    flat_merged_matrix = merged_matrix.flatten().astype(int)
    
    # clean up the neighbors and weights matrix by considering only those idx values in the 
    # merged_matrix (this is removing padded elements from the dicts)
    neighbors_dict, weights_dict = _remove_padded_elements_from_weights_neigbors_dicts(
        neighbors_dict, weights_dict, flat_merged_matrix
    )

    return {
        'neighbors':neighbors_dict, 
        'weights':weights_dict
    }
    