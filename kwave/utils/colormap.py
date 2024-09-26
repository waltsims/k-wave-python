from typing import Optional

import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Float
from matplotlib.colors import ListedColormap


@typechecker
def get_color_map(num_colors: Optional[int] = None) -> ListedColormap:
    """
    Returns the default color map used for display and visualisation across
    the k-Wave Toolbox. Zero values are displayed as white, positive values
    are displayed as yellow through red to black, and negative values are
    displayed as light to dark blue-greys. If no value for `num_colors` is
    provided, `cm` will have 256 colors.

    Args:
        num_colors: The number of colors in the color map (default is 256).

    Returns:
        A three-column color map matrix which can be applied using colormap.

    """

    if num_colors is None:
        neg_pad = 48
        num_colors = 256
    else:
        neg_pad = int(round(48 * num_colors / 256))

    # define colour spectrums
    neg = bone(num_colors // 2 + neg_pad)
    neg = neg[neg_pad:, :]
    pos = np.flipud(hot(num_colors // 2))

    colors = np.vstack([neg, pos])
    return ListedColormap(colors)


@typechecker
def hot(m: int) -> Float[np.ndarray, "N 3"]:
    """
    Generate a hot colormap of length m.
    The colormap consists of a progression from black to red, yellow, and white.

    Args:
        m: The length of the colormap.

    Returns:
        An m-by-3 array containing the hot colormap.

    """

    n = int(np.fix(3 / 8 * m))

    r = np.concatenate([np.arange(1, n + 1) / n, np.ones(m - n)])
    g = np.concatenate([np.zeros(n), np.arange(1, n + 1) / n, np.ones(m - 2 * n)])
    b = np.concatenate([np.zeros(2 * n), np.arange(1, m - 2 * n + 1) / (m - 2 * n)])

    return np.hstack([r[:, None], g[:, None], b[:, None]])


@typechecker
def bone(m: int) -> Float[np.ndarray, "N 3"]:
    """
    Returns an m-by-3 matrix containing a "bone" colormap.

    Args:
        m: The number of rows in the colormap.

    Returns:
        An m-by-3 matrix containing the colormap.
    """
    return (7 * gray(m) + np.fliplr(hot(m))) / 8


@typechecker
def gray(m: int) -> Float[np.ndarray, "N 3"]:
    """
    Returns an M-by-3 matrix containing a grayscale colormap.

    Args:
        m: The length of the colormap.

    Returns:
        An M-by-3 matrix containing the grayscale colormap.

    """

    g = np.arange(m) / max(m - 1, 1)
    g = g[:, None]
    return np.hstack([g, g, g])
