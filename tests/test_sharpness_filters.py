import numpy as np

from kwave.utils.sharpness_filters import brenner_sharpness


class TestBrennerSharpness:
    """Tests for the brenner_sharpness function."""

    def test_2d_constant_image(self):
        """Test brenner_sharpness with a 2D constant image (should return 0)."""
        im = np.ones((10, 10))
        result = brenner_sharpness(im)
        assert result == 0

    def test_2d_gradient_image(self):
        """Test brenner_sharpness with a 2D gradient image."""
        im = np.zeros((10, 10))
        # Add horizontal gradien
        for i in range(10):
            im[:, i] = i
        result = brenner_sharpness(im)
        assert result > 0

    def test_2d_edge_image(self):
        """Test brenner_sharpness with a 2D image containing an edge."""
        im = np.zeros((10, 10))
        im[:, 5:] = 1  # Sharp edge in the middle
        result = brenner_sharpness(im)
        assert result > 0

    def test_3d_constant_image(self):
        """Test brenner_sharpness with a 3D constant image (should return 0)."""
        im = np.ones((5, 5, 5))
        result = brenner_sharpness(im)
        assert result == 0

    def test_3d_gradient_image(self):
        """Test brenner_sharpness with a 3D gradient image."""
        im = np.zeros((5, 5, 5))
        # Add gradient in z direction
        for i in range(5):
            im[:, :, i] = i
        result = brenner_sharpness(im)
        assert result > 0

    def test_3d_edge_image(self):
        """Test brenner_sharpness with a 3D image containing edges."""
        im = np.zeros((5, 5, 5))
        im[:, :, 2:] = 1  # Sharp edge in the middle along z-axis
        result = brenner_sharpness(im)
        assert result > 0

    def test_minimum_size_2d(self):
        """Test brenner_sharpness with a minimum size 2D image that still allows calculation."""
        # Minimum size is 3x3 due to the [-2:] and [2:] indexing
        im = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        result = brenner_sharpness(im)
        assert result >= 0

    def test_minimum_size_3d(self):
        """Test brenner_sharpness with a minimum size 3D image that still allows calculation."""
        # Minimum size is 3x3x3 due to the [-2:] and [2:] indexing
        im = np.zeros((3, 3, 3))
        im[:, :, 2] = 1  # Set the last slice to 1
        result = brenner_sharpness(im)
        assert result >= 0

    def test_relative_sharpness_2d(self):
        """Test that a sharper 2D image gives higher brenner_sharpness value."""
        # Create a blurry edge
        im_blurry = np.zeros((10, 10))
        im_blurry[:, 3] = 0.25
        im_blurry[:, 4] = 0.5
        im_blurry[:, 5] = 0.75
        im_blurry[:, 6:] = 1.0

        # Create a sharp edge
        im_sharp = np.zeros((10, 10))
        im_sharp[:, 5:] = 1.0

        blurry_result = brenner_sharpness(im_blurry)
        sharp_result = brenner_sharpness(im_sharp)

        assert sharp_result > blurry_result

    def test_relative_sharpness_3d(self):
        """Test that a sharper 3D image gives higher brenner_sharpness value."""
        # Create a blurry edge
        im_blurry = np.zeros((5, 5, 5))
        im_blurry[:, :, 1] = 0.33
        im_blurry[:, :, 2] = 0.67
        im_blurry[:, :, 3:] = 1.0

        # Create a sharp edge
        im_sharp = np.zeros((5, 5, 5))
        im_sharp[:, :, 3:] = 1.0

        blurry_result = brenner_sharpness(im_blurry)
        sharp_result = brenner_sharpness(im_sharp)

        assert sharp_result > blurry_result
