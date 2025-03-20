from typing import Optional

import numpy as np
from scipy.ndimage import convolve


def brenner_sharpness(im):
    """Calculate Brenner's measure of focus/sharpness.

    Computes squared differences between pixels that are 2 units apart
    along each dimension, then sums all differences.

    Args:
        im: Input image/volume array

    Returns:
        Single scalar value representing the sharpness measure
    """
    s = 0
    for dim in range(im.ndim):
        # Create slices for the current dimension
        slice1 = [slice(None)] * im.ndim
        slice2 = [slice(None)] * im.ndim

        # Set the slices for the dimension we're processing
        slice1[dim] = slice(None, -2)
        slice2[dim] = slice(2, None)

        # Calculate squared difference for this dimension and add to total
        brenner_dim = (im[tuple(slice1)] - im[tuple(slice2)]) ** 2
        s += np.sum(brenner_dim)

    return s


def tenenbaum_sharpness(im: np.ndarray) -> float | None:
    num_dim = im.ndim
    if num_dim == 2:
        # define the 2D sobel gradient operator
        sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # compute metric
        s = (convolve(sobel, im) ** 2 + convolve(sobel.T, im) ** 2).sum()
    elif num_dim == 3:
        # define the 3D sobel gradient operator
        sobel3D = np.zeros((3, 3, 3))
        sobel3D[:, :, 0] = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        sobel3D[:, :, 2] = -sobel3D[:, :, 0]

        # compute metric
        s = (
            convolve(im, sobel3D) ** 2
            + convolve(im, np.transpose(sobel3D, (2, 0, 1))) ** 2
            + convolve(im, np.transpose(sobel3D, (1, 2, 0))) ** 2
        ).sum()
    return s

    # TODO: get this passing the tests
    # NOTE: Walter thinks this is the proper way to do this, but it doesn't match the MATLAB version
    # num_dim = im.ndim
    # if num_dim == 2:
    #     # compute metric
    #     sx = sobel(im, axis=0, mode='constant')
    #     sy = sobel(im, axis=1, mode='constant')
    #     s = (sx ** 2) + (sy ** 2)
    #     s = np.sum(s)
    #
    # elif num_dim == 3:
    #     # compute metric
    #     sx = sobel(im, axis=0, mode='constant')
    #     sy = sobel(im, axis=1, mode='constant')
    #     sz = sobel(im, axis=2, mode='constant')
    #     s = (sx ** 2) + (sy ** 2) + (sz ** 2)
    #     s = np.sum(s)
    # else:
    #     raise ValueError("Invalid number of dimensions in im")


def norm_var(im: np.ndarray) -> float:
    """
    Calculates the normalized variance of an array of values.

    Args:
        im: The input array.

    Returns:
        The normalized variance of im.

    """
    mu = np.mean(im)
    s = np.sum((im - mu) ** 2) / mu
    return s


def sharpness(im: np.ndarray, mode: Optional[str] = "Brenner") -> float:
    """
    Returns a scalar metric related to the sharpness of a 2D or 3D image matrix.

    Args:
        im: The image matrix.
        metric: The metric to use. Defaults to "Brenner".

    Returns:
        A scalar sharpness metric.

    Raises:
        AssertionError: If `im` is not a NumPy array.

    References:
        B. E. Treeby, T. K. Varslot, E. Z. Zhang, J. G. Laufer, and P. C. Beard, "Automatic sound speed selection in
        photoacoustic image reconstruction using an autofocus approach," J. Biomed. Opt., vol. 16, no. 9, p. 090501, 2011.

    """

    assert isinstance(im, np.ndarray), "Argument im must be of type numpy array"

    if mode == "Brenner":
        metric = brenner_sharpness(im)
    elif mode == "Tenenbaum":
        metric = tenenbaum_sharpness(im)
    elif mode == "NormVariance":
        metric = norm_var(im)
    else:
        raise ValueError("Unrecognized sharpness metric passed. " "Valid values are ['Brenner', 'Tanenbaum', 'NormVariance']")

    return metric
