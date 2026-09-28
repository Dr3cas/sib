import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.metrics.accuracy import accuracy
from si.metrics.rmse import rmse
from si.model_selection.split import stratified_train_test_split, train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.knn_regressor import KNNRegressor


class TestMetrics(TestCase):

    def test_accuracy(self):
        self.assertEqual(0.75, accuracy(np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1])))

    def test_rmse(self):
        # errors: -1, -1, 2, -1 -> squares 1, 1, 4, 1 -> mean 1.75
        self.assertAlmostEqual(np.sqrt(1.75), rmse(np.array([1., 2., 3., 4.]), np.array([2., 3., 1., 5.])))
        self.assertAlmostEqual(0.0, rmse(np.array([1., 2.]), np.array([1., 2.])))


class TestKNNClassifier(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_predict_and_score(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)
        knn = KNNClassifier(k=3).fit(train)
        predictions = knn.predict(test)

        self.assertEqual(test.shape()[0], len(predictions))
        # classes must not be truncated (e.g., 'Iris-versic')
        self.assertTrue(set(predictions).issubset(set(self.dataset.get_classes())))
        self.assertEqual(accuracy(test.y, predictions), knn.score(test))
        self.assertGreater(knn.score(test), 0.9)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            KNNClassifier(k=3).predict(self.dataset)

    def test_score_before_fit_raises(self):
        with self.assertRaises(ValueError):
            KNNClassifier(k=3).score(self.dataset)


class TestKNNRegressor(TestCase):

    def test_predict_and_score(self):
        cpu = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features=True, label=True)
        train, test = train_test_split(cpu, test_size=0.2, random_state=42)
        knn = KNNRegressor(k=3).fit(train)
        predictions = knn.predict(test)

        self.assertEqual(test.shape()[0], len(predictions))
        self.assertAlmostEqual(rmse(test.y, predictions), knn.score(test))

    def test_average_of_neighbours(self):
        X = np.array([[0.], [1.], [2.], [10.]])
        y = np.array([10., 20., 30., 100.])
        knn = KNNRegressor(k=2).fit(Dataset(X, y))
        predictions = knn.predict(Dataset(np.array([[0.4]])))
        np.testing.assert_allclose(predictions, [15.0])
