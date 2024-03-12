import matplotlib.pyplot as plt
import numpy as np


def voxel_plot(mat, axis_tight=False, color=(1, 1, 0.4), transparency=0.8):
    """
    Generates a 3D voxel plot of a binary matrix.

    Args:
        mat (numpy.ndarray): Input 3D matrix in single or double precision.
        axis_tight (bool): Whether axis limits are set to only display the filled voxels (default = False).
        color (tuple): Three-element tuple specifying RGB color (default = (1, 1, 0.4)).
        transparency (float): Value between 0 and 1 specifying transparency where 1 gives no transparency (default = 0.8).

    Returns:
        None
    """
    # Check input matrix is 3D and single or double precision
    if len(mat.shape) != 3 or not np.issubdtype(mat.dtype, np.floating):
        raise ValueError("Input must be a 3D matrix in single or double precision.")

    # Normalize the matrix
    mat = mat / np.max(mat)

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(mat, facecolors=color, alpha=transparency, edgecolors=(0.5, 0.5, 0.5, 0.5))

    # Set the axes properties and labels
    ax.view_init(elev=35, azim=-35)  # Adjust viewing angles here
    ax.set_xlabel("x [voxels]")
    ax.set_ylabel("y [voxels]")
    ax.set_zlabel("z [voxels]")

    if not axis_tight:
        sz = mat.shape
        ax.set_xlim([0.5, sz[2] + 0.5])
        ax.set_ylim([0.5, sz[1] + 0.5])
        ax.set_zlim([0.5, sz[0] + 0.5])

    # Show the plot
    plt.show()
