from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and testing sets (random split).

    Parameters
    ----------
    dataset: Dataset
        The Dataset object to split into training and testing data.
    test_size: float
        The size of the testing Dataset (e.g., 0.2 for 20%).
    random_state: int
        Seed for generating permutations.

    Returns
    -------
    train, test: Tuple[Dataset, Dataset]
        A tuple containing the train and test Dataset objects.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.shape()[0]
    n_test = int(n_samples * test_size)

    permutations = np.random.permutation(n_samples)
    test_idxs = permutations[:n_test]
    train_idxs = permutations[n_test:]

    train = Dataset(dataset.X[train_idxs],
                    dataset.y[train_idxs] if dataset.has_label() else None,
                    features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs],
                   dataset.y[test_idxs] if dataset.has_label() else None,
                   features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2,
                                random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and testing sets keeping the proportion
    of each class (stratified split).

    Parameters
    ----------
    dataset: Dataset
        The Dataset object to split into training and testing data.
    test_size: float
        The size of the testing Dataset (e.g., 0.2 for 20%).
    random_state: int
        Seed for generating permutations.

    Returns
    -------
    train, test: Tuple[Dataset, Dataset]
        A tuple containing the stratified train and test Dataset objects.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # unique class labels and their counts
    unique_labels, counts = np.unique(dataset.y, return_counts=True)

    train_idxs = []
    test_idxs = []

    for label, count in zip(unique_labels, counts):
        # number of test samples for the current class
        n_test_class = int(count * test_size)

        # shuffle the indices of the current class and select the test ones
        class_idxs = np.where(dataset.y == label)[0]
        np.random.shuffle(class_idxs)

        test_idxs.extend(class_idxs[:n_test_class])
        train_idxs.extend(class_idxs[n_test_class:])

    train_idxs = np.array(train_idxs)
    test_idxs = np.array(test_idxs)

    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs],
                    features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs],
                   features=dataset.features, label=dataset.label)
    return train, test
