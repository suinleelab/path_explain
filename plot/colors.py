"""
A place for defining the default color scheme.
"""

import numpy as np
import matplotlib as mpl

def green_gold():
    """
    Returns the green and gold colormap we use as the
    default color scheme for this repository.
    """
    color_map_size = 256
    vals = np.ones((color_map_size, 4))
    vals[:, 0] = np.linspace(250/256, 20/256, color_map_size)
    vals[:, 1] = np.linspace(230/256, 125/256, color_map_size)
    vals[:, 2] = np.linspace(0/256, 0/256, color_map_size)
    cmap = mpl.colors.ListedColormap(vals)
    return cmap
