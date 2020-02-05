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
    vals[:, 0] = np.linspace(20/256, 250/256, color_map_size)
    vals[:, 1] = np.linspace(125/256, 230/256, color_map_size)
    vals[:, 2] = np.linspace(0/256, 0/256, color_map_size)
    cmap = mpl.colors.ListedColormap(vals)
    return cmap

def maroon_white_aqua():
    """
    Returns the green and gold colormap we use as the
    default color scheme for plotting text.
    """
    color_map_size = 256
    vals = np.ones((color_map_size, 4))
    vals[:int(color_map_size / 2), 0] = np.linspace(140/256, 1.0, int(color_map_size / 2))
    vals[:int(color_map_size / 2), 1] = np.linspace(15/256, 1.0, int(color_map_size / 2))
    vals[:int(color_map_size / 2), 2] = np.linspace(15/256, 1.0, int(color_map_size / 2))

    vals[int(color_map_size / 2):, 0] = np.linspace(1.0, 0/256, int(color_map_size / 2))
    vals[int(color_map_size / 2):, 1] = np.linspace(1.0, 220/256, int(color_map_size / 2))
    vals[int(color_map_size / 2):, 2] = np.linspace(1.0, 170/256, int(color_map_size / 2))
    cmap = mpl.colors.ListedColormap(vals)
    return cmap
