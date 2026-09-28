import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    # ------------------------------------------------------------------
    # Exercise 2: dropna, fillna and remove_by_index
    # ------------------------------------------------------------------
    def setUp(self):
        self.X = np.array([[1.0, 2.0, np.nan],
                           [4.0, np.nan, 6.0],
                           [7.0, 8.0, 9.0]])
        self.y = np.array([0, 1, 0])
        self.features = ['f1', 'f2', 'f3']
        self.label = 'y'

    def _dataset_with_nans(self):
        return Dataset(self.X.copy(), self.y.copy(), self.features, self.label)

    def test_dropna(self):
        dataset = self._dataset_with_nans()
        result = dataset.dropna()

        self.assertIs(result, dataset)
        self.assertEqual((1, 3), dataset.shape())
        self.assertEqual(1, dataset.y.shape[0])
        self.assertFalse(np.isnan(dataset.X).any())
        np.testing.assert_array_equal([0], dataset.y)

    def test_fillna_value(self):
        dataset = self._dataset_with_nans()
        result = dataset.fillna(0)

        self.assertIs(result, dataset)
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(0, dataset.X[0, 2])
        self.assertEqual(0, dataset.X[1, 1])
        self.assertEqual((3, 3), dataset.shape())

    def test_fillna_mean(self):
        dataset = self._dataset_with_nans()
        dataset.fillna('mean')

        self.assertFalse(np.isnan(dataset.X).any())
        self.assertAlmostEqual(np.nanmean(self.X[:, 1]), dataset.X[1, 1])
        self.assertAlmostEqual(np.nanmean(self.X[:, 2]), dataset.X[0, 2])

    def test_fillna_median(self):
        dataset = self._dataset_with_nans()
        dataset.fillna('median')

        self.assertFalse(np.isnan(dataset.X).any())
        self.assertAlmostEqual(np.nanmedian(self.X[:, 1]), dataset.X[1, 1])
        self.assertAlmostEqual(np.nanmedian(self.X[:, 2]), dataset.X[0, 2])

    def test_fillna_invalid_string(self):
        dataset = self._dataset_with_nans()
        with self.assertRaises(ValueError):
            dataset.fillna('banana')

    def test_remove_by_index(self):
        dataset = self._dataset_with_nans()
        result = dataset.remove_by_index(0)

        self.assertIs(result, dataset)
        self.assertEqual((2, 3), dataset.shape())
        np.testing.assert_array_equal([1, 0], dataset.y)
        self.assertEqual(4.0, dataset.X[0, 0])

