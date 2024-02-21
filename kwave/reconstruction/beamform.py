import logging
from typing import Tuple, Optional
import numpy as np
import scipy

from kwave.utils.conversion import cart2pol
from kwave.utils.data import scale_time
from kwave.utils.tictoc import TicToc


def focus(kgrid, input_signal, source_mask, focus_position, sound_speed):
    """
    focus Create input signal based on source mask and focus position.
    focus takes a single input signal and a source mask and creates an
    input signal matrix (with one input signal for each source point).
    The appropriate time delays required to focus the signals at a given
    position in Cartesian space are automatically added based on the user
    inputs for focus_position and sound_speed.

    Args:
         kgrid:             k-Wave grid object returned by kWaveGrid
         input_signal:      single time series input
         source_mask:       matrix specifying the positions of the time
                            varying source distribution (i.e., source.p_mask
                            or source.u_mask)
         focus_position:    position of the focus in Cartesian coordinates
         sound_speed:       scalar sound speed

    Returns:
         input_signal_mat:  matrix of time series following the source points
    """

    assert not isinstance(kgrid.t_array, str), "kgrid.t_array must be a numeric array."

    if isinstance(sound_speed, int):
        sound_speed = float(sound_speed)

    assert isinstance(sound_speed, float), "sound_speed must be a scalar."

    positions = [kgrid.x.flatten(), kgrid.y.flatten(), kgrid.z.flatten()]

    # filter_positions
    positions = [position for position in positions if (position != np.nan).any()]
    assert len(positions) == kgrid.dim
    positions = np.array(positions)

    if isinstance(focus_position, list):
        focus_position = np.array(focus_position)
    assert isinstance(focus_position, np.ndarray)

    dist = np.linalg.norm(positions[:, source_mask.flatten() == 1] - focus_position[:, np.newaxis])

    # distance to delays
    delay = int(np.round(dist / (kgrid.dt * sound_speed)))
    max_delay = np.max(delay)
    rel_delay = -(delay - max_delay)

    signal_mat = np.zeros((rel_delay.size, input_signal.size + max_delay))

    # for src_idx, delay in enumerate(rel_delay):
    #     signal_mat[src_idx, delay:max_delay - delay] = input_signal
    # signal_mat[rel_delay, delay:max_delay - delay] = input_signal

    logging.log(
        logging.WARN, f"{PendingDeprecationWarning.__name__}: " "This method is not fully migrated, might be depricated and is untested."
    )

    return signal_mat


def scan_conversion(
    scan_lines: np.ndarray, steering_angles, image_size: Tuple[float, float], c0, dt, resolution: Optional[Tuple[int, int]]
) -> np.ndarray:
    if resolution is None:
        resolution = (256, 256)  # in pixels

    x_resolution, y_resolution = resolution

    # assign the inputs
    x, y = image_size

    # start the timer
    TicToc.tic()

    # update command line status
    logging.log(logging.INFO, "Computing ultrasound scan conversion...")

    # extract a_line parameters
    Nt = scan_lines.shape[1]

    # calculate radius variable based on the sound speed in the medium and the
    # round trip distance
    r = c0 * np.arange(1, Nt + 1) * dt / 2  # [m]

    # create regular Cartesian grid to remap to
    pos_vec_y_new = np.linspace(0, 1, y_resolution) * y - y / 2
    pos_vec_x_new = np.linspace(0, 1, x_resolution) * x
    [pos_mat_x_new, pos_mat_y_new] = np.array(np.meshgrid(pos_vec_x_new, pos_vec_y_new, indexing="ij"))

    # convert new points to polar coordinates
    [th_cart, r_cart] = cart2pol(pos_mat_x_new, pos_mat_y_new)

    # TODO: move this import statement at the top of the file
    # Not possible now due to cyclic dependencies
    from kwave.utils.interp import interpolate2d_with_queries

    # below part has some modifications
    # we flatten the _cart matrices and build queries
    # then we get values at the query locations
    # and reshape the values to the desired size
    # These three steps can be accomplished in one step in Matlab
    # However, we don't want to add custom logic to the `interpolate2D_with_queries` method.

    # Modifications -start
    queries = np.array([r_cart.flatten(), th_cart.flatten()]).T

    b_mode = interpolate2d_with_queries([r, 2 * np.pi * steering_angles / 360], scan_lines.T, queries, method="linear", copy_nans=False)
    image_size_points = (len(pos_vec_x_new), len(pos_vec_y_new))
    b_mode = b_mode.reshape(image_size_points)
    # Modifications -end

    b_mode[np.isnan(b_mode)] = 0

    # update command line status
    logging.log(logging.INFO, f"  completed in {scale_time(TicToc.toc())}")

    return b_mode


def envelope_detection(signal):
    """
    envelopeDetection applies the Hilbert transform to extract the
    envelope from an input vector x. If x is a matrix, the envelope along
    the last axis.

    Args:
        signal:

    Returns:
        signal_envelope:

    """

    return np.abs(scipy.signal.hilbert(signal))
