import numpy as np


def get_color_map(num_colors=None):
    """
      DESCRIPTION:
          getColorMap returns the default color map used for display and
          visualisation across the k-Wave Toolbox. Zero values are displayed as
          white, positive values are displayed as yellow through red to black,
          and negative values are displayed as light to dark blue-greys. If no
          value for num_colors is provided, cm will have 256 colors.
    Args:
        num_colors: number of colors in the color map (default is 256)

    Returns:
        cm: three column color map matrix which can be applied using colormap
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

    return np.vstack([neg, pos])


def hot(m):
    """
    %HOT    Red-yellow-white color map inspired by black body radiation
    %   HOT(M) returns an M-by-3 matrix containing a "hot" colormap.
    %   HOT, by itself, is the same length as the current figure's
    %   colormap. If no figure exists, MATLAB uses the length of the
    %   default colormap.
    Args:
        m:

    Returns:

    """
    n = int(np.fix(3/8 * m))

    r = np.concatenate([np.arange(1, n + 1) / n, np.ones(m-n)])
    g = np.concatenate([np.zeros(n), np.arange(1, n + 1) / n, np.ones(m-2*n)])
    b = np.concatenate([np.zeros(2*n), np.arange(1, m-2*n + 1)/(m-2*n)])

    return np.hstack([r[:, None], g[:, None], b[:, None]])


def bone(m):
    return (7 * gray(m) + np.fliplr(hot(m))) / 8


def gray(m):
    g = np.arange(m) / max(m-1, 1)
    g = g[:, None]
    return np.hstack([g, g, g])