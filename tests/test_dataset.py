import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1.0, 2.0, np.nan],
                            [4.0, np.nan, 6.0],
                            [7.0, 8.0, 9.0]])
        self.y = np.array([0, 1, 0])
        self.features = ["f1", "f2", "f3"]
        self.label = "y"

    def test_shape(self):
        ds = Dataset(X=self.X, y=self.y, features=self.features, label=self.label)
        self.assertEqual(ds.shape, (3, 3))

    def test_has_label(self):
        ds = Dataset(X=self.X, y=self.y)
        self.assertTrue(ds.has_label())

        ds_no_label = Dataset(X=self.X)
        self.assertFalse(ds_no_label.has_label())

    def test_get_classes(self):
        ds = Dataset(X=self.X, y=self.y)
        np.testing.assert_array_equal(ds.get_classes(), [0, 1])

    def test_dropna(self):
        ds = Dataset(X=self.X.copy(), y=self.y.copy(),
                      features=self.features, label=self.label)
        ds.dropna()
        self.assertEqual(ds.shape[0], 1)
        self.assertEqual(ds.y.shape[0], 1)
        self.assertFalse(np.isnan(ds.X).any())

    def test_fillna_value(self):
        ds = Dataset(X=self.X.copy(), y=self.y.copy(),
                      features=self.features, label=self.label)
        ds.fillna(0)
        self.assertFalse(np.isnan(ds.X).any())
        self.assertEqual(ds.X[0, 2], 0)

    def test_fillna_mean(self):
        ds = Dataset(X=self.X.copy(), y=self.y.copy(),
                      features=self.features, label=self.label)
        expected_mean_f2 = np.nanmean(self.X[:, 1])
        ds.fillna("mean")
        self.assertFalse(np.isnan(ds.X).any())
        self.assertAlmostEqual(ds.X[1, 1], expected_mean_f2)

    def test_fillna_median(self):
        ds = Dataset(X=self.X.copy(), y=self.y.copy(),
                      features=self.features, label=self.label)
        expected_median_f2 = np.nanmedian(self.X[:, 1])
        ds.fillna("median")
        self.assertFalse(np.isnan(ds.X).any())
        self.assertAlmostEqual(ds.X[1, 1], expected_median_f2)

    def test_remove_by_index(self):
        ds = Dataset(X=self.X.copy(), y=self.y.copy(),
                      features=self.features, label=self.label)
        ds.remove_by_index(0)
        self.assertEqual(ds.shape[0], 2)
        self.assertEqual(ds.y.shape[0], 2)
        np.testing.assert_array_equal(ds.y, [1, 0])


if __name__ == "__main__":
    unittest.main()
