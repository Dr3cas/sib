import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.data.dataset import Dataset
from si.feature_selection.select_k_best import SelectKBest
from si.feature_selection.variance_threshold import VarianceThreshold
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification


class TestVarianceThreshold(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_fit_estimates_variance(self):
        selector = VarianceThreshold(threshold=0.5).fit(self.dataset)
        np.testing.assert_allclose(selector.variance, np.var(self.dataset.X, axis=0))

    def test_transform_iris(self):
        transformed = VarianceThreshold(threshold=0.5).fit_transform(self.dataset)
        self.assertEqual((150, 3), transformed.shape())
        self.assertEqual(['sepal_length', 'petal_length', 'petal_width'], transformed.features)

    def test_constant_feature_is_removed(self):
        dataset = Dataset.from_random(100, 10)
        dataset.X[:, 2] = 0
        transformed = VarianceThreshold().fit_transform(dataset)
        self.assertEqual((100, 9), transformed.shape())
        self.assertNotIn('feat_2', transformed.features)


class TestSelectKBest(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_f_classification(self):
        F, p = f_classification(self.dataset)
        self.assertEqual(4, len(F))
        self.assertEqual(4, len(p))
        self.assertTrue(np.all(p < 0.05))

    def test_fit_estimates_f_and_p(self):
        selector = SelectKBest(score_func=f_classification, k=2).fit(self.dataset)
        self.assertEqual(4, len(selector.F))
        self.assertEqual(4, len(selector.p))

    def test_transform_iris(self):
        transformed = SelectKBest(score_func=f_classification, k=2).fit_transform(self.dataset)
        self.assertEqual((150, 2), transformed.shape())
        self.assertEqual(['petal_length', 'petal_width'], transformed.features)
        self.assertTrue(transformed.has_label())
