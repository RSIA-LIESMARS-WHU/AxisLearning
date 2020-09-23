import numpy as np
cimport numpy as np

cdef extern from "rbbox_overlaps.hpp":
    void _overlaps(np.float32_t*, np.float32_t*, np.float32_t*, int, int, int)




def rbox_2_locations (np.ndarray[np.float32_t, ndim=2] boxes, np.ndarray[np.float32_t, ndim=2] mutiFeaturesPts, np.int32_t device_id=0):
    cdef int N = boxes.shape[0]
    cdef int K = mutiFeaturesPts.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] gridpts2targets = np.zeros((K, 8), dtype = np.float32)
    _overlaps(&gridpts2targets[0, 0], &boxes[0, 0], &mutiFeaturesPts[0, 0], N, K, device_id)
    return gridpts2targets
