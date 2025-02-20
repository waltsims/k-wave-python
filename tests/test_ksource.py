import unittest
import numpy as np
from kwave.ksource import kSource


class TestKSource(unittest.TestCase):
    def setUp(self):
        self.source = kSource()

    def test_p0_setter_empty_array(self):
        """Test that p0 is set to None when given an empty array"""
        self.source.p0 = np.array([])
        self.assertIsNone(self.source.p0)

    def test_p0_setter_non_empty_array(self):
        """Test that p0 is set correctly for non-empty array"""
        test_array = np.array([1.0, 2.0, 3.0])
        self.source.p0 = test_array
        np.testing.assert_array_equal(self.source.p0, test_array)

    def test_p0_setter_zero_array(self):
        """Test that p0 is set correctly for array of zeros (should not be set to None)"""
        test_array = np.zeros(5)
        self.source.p0 = test_array
        np.testing.assert_array_equal(self.source.p0, test_array)

    def test_is_p0_empty(self):
        """Test the is_p0_empty method"""
        # Test with None
        self.assertTrue(self.source.is_p0_empty())

        # Test with empty array
        self.source.p0 = np.array([])
        self.assertTrue(self.source.is_p0_empty())

        # Test with non-empty array
        self.source.p0 = np.array([1.0, 2.0])
        self.assertFalse(self.source.is_p0_empty())

        # Test with zero array
        self.source.p0 = np.zeros(5)
        self.assertTrue(self.source.is_p0_empty())
