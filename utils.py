import numpy as np

def normalize(im_batch, _range=None, _domain=None):
    if len(im_batch.shape) == 2:
        axis = (0, 1)
    elif len(im_batch.shape) == 3:
        axis = (0, 1, 2)
    elif len(im_batch.shape) == 4:
        axis = (1, 2, 3)
    else:
        raise ValueError('im_batch must be of rank 2, 3 or 4')
    
    if _domain is not None:
        min_vals = _domain[0]
        max_vals = _domain[1]
    else:
        min_vals = np.amin(im_batch, axis=axis, keepdims=True)
        max_vals = np.amax(im_batch, axis=axis, keepdims=True)
    
    norm_batch = (im_batch - min_vals) / (max_vals - min_vals)
    
    if _range is not None:
        amin = _range[0]
        amax = _range[1]
        norm_batch = norm_batch * (amax - amin) + amin
    return norm_batch