import numpy as np
from skimage.morphology import disk

####################
# Background constants
####################
# size of kernel used to smooth estimated background
# ~ 1/4 size of frame in each dimension
XENOPUS_BCKGR_GAUS_KERNEL = (259, 347)

# intensity percentile to take over all frames in a video
XENOPUS_BCKGR_PERCENTILE = 2

XENOPUS_BCKGR_PATCH_SHAPE = (130, 174)

####################
# Segment constants
####################
# fraction of max distance transform to use as threhsold
DISTANCE_TRANSFORM_THRESH = 0.7

# min and max areas (in pixels) of a single embryo
MIN_EMBRYO_AREA = 17000
MAX_EMBRYO_AREA = 30000

# block size to use in the adaptive threshold algorithm
ADAPTIVE_THRESH_BLOCK_SIZE = 253

# kernel sizes for morphological operations
OPEN_KERNEL_SIZE = (11, 11)
DILATE_KERNEL_SIZE = (11, 11)


####################
# Gradient segmentation constants
####################

SCHARR_KERNEL = np.array([
    [-3-3j, 0-10j, 3-3j],
    [-10+0j, 0+0j, 10+0j],
    [-3+3j, 0+10j, 3+3j]
])

GRAD_MIN_AREA = 10000

####################
# Colour segmentation constants
####################

HSV_MIN_THRESHOLD = np.array([12, 70, 62])

####################
# HBEC consts
####################

HBEC_ROI_RADIUS = 990
HBEC_WINDOW_SIZE = (32, 32)
HBEC_STEPS_SIZE = (4, 4)
HBEC_RATIO_PER_PATCH_THRESH = 0.3
