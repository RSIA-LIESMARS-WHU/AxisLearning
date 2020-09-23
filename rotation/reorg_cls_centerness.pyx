import numpy as np
cimport numpy as np

cdef extern from "reorg_cls_centerness.hpp":
    void _reorg_feature(np.float32_t*, np.float32_t*, int, int, int, int)

def reorg_cls_center (np.ndarray[np.float32_t, ndim=2] ori_cls_centerness,
                  np.int32_t height, 
                  np.int32_t width, 
                  np.int32_t num_cls, 
                  np.int32_t device_id=0):
    cdef int K = ori_cls_centerness.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] reorg_cls_centerness = np.zeros((K, num_cls+1), dtype = np.float32)
    _reorg_feature(&reorg_cls_centerness[0, 0], &ori_cls_centerness[0, 0], height, width, num_cls, device_id)
    return reorg_cls_centerness


