from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
import os

from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv


class TestSelectPercentile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(self.csv_file, features=True, label=True)

    def test_transform_slide_example(self):
        # example from the slide: 10 features, percentile=40 (ties at the threshold)
        F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
        dataset = Dataset(np.zeros((3, 10)), np.array([0, 1, 0]), features=[str(v) for v in F])

        select = SelectPercentile(percentile=40)
        select.F = F
        transformed = select._transform(dataset)

        self.assertEqual((3, 4), transformed.shape())
        self.assertEqual([5.6, 5.6, 7.8, 6.5], [float(f) for f in transformed.features])

    def test_fit_transform_iris(self):
        select = SelectPercentile(percentile=50)
        transformed = select.fit_transform(self.dataset)

        self.assertEqual((150, 2), transformed.shape())
        self.assertEqual(['petal_length', 'petal_width'], transformed.features)
        self.assertEqual(4, len(select.F))
        self.assertEqual(4, len(select.p))

    def test_percentile_100_keeps_all_features(self):
        transformed = SelectPercentile(percentile=100).fit_transform(self.dataset)
        self.assertEqual((150, 4), transformed.shape())

    def test_percentile_25_keeps_best_feature(self):
        transformed = SelectPercentile(percentile=25).fit_transform(self.dataset)
        self.assertEqual((150, 1), transformed.shape())
        self.assertEqual(['petal_length'], transformed.features)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=50).transform(self.dataset)
