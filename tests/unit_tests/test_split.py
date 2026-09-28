import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split, stratified_train_test_split


class TestSplit(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_train_test_split(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=123)
        self.assertEqual(30, test.shape()[0])
        self.assertEqual(120, train.shape()[0])
        self.assertEqual(4, train.shape()[1])
        self.assertEqual(train.shape()[0], len(train.y))

    def test_train_test_split_is_reproducible(self):
        train1, test1 = train_test_split(self.dataset, test_size=0.2, random_state=42)
        train2, test2 = train_test_split(self.dataset, test_size=0.2, random_state=42)
        np.testing.assert_array_equal(train1.X, train2.X)
        np.testing.assert_array_equal(test1.X, test2.X)

    def test_train_test_split_does_not_mix_samples(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=1)
        all_X = np.vstack((train.X, test.X))
        self.assertEqual(150, all_X.shape[0])
        # every original sample is used exactly once (iris has a few duplicated rows, so compare sums)
        self.assertAlmostEqual(self.dataset.X.sum(), all_X.sum())

    def test_stratified_train_test_split(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        self.assertEqual((30, 4), test.shape())
        self.assertEqual((120, 4), train.shape())
        # 50 samples per class -> 10 per class in test and 40 per class in train
        for dataset, n in ((test, 10), (train, 40)):
            classes, counts = np.unique(dataset.y, return_counts=True)
            self.assertEqual(3, len(classes))
            self.assertTrue(np.all(counts == n))
