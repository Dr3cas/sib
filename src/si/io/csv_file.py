import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False) -> Dataset:
    """
    Reads a csv file and returns a Dataset object.

    Parameters
    ----------
    filename: str
        Name/path of the file.
    sep: str
        Value separator.
    features: bool
        Whether the file has feature names.
    label: bool
        Whether the file has a label (y). If True, it is assumed to be the last column.

    Returns
    -------
    Dataset
    """
    data = pd.read_csv(filename, sep=sep)

    if features and label:
        feature_names = data.columns[:-1].tolist()
        label_name = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
    elif features and not label:
        feature_names = data.columns.tolist()
        label_name = None
        X = data.to_numpy()
        y = None
    elif not features and label:
        feature_names = None
        label_name = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
    else:
        feature_names = None
        label_name = None
        X = data.to_numpy()
        y = None

    return Dataset(X, y, features=feature_names, label=label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file.

    Parameters
    ----------
    filename: str
        Name/path of the file.
    dataset: Dataset
        The dataset to write.
    sep: str
        Value separator.
    features: bool
        Whether to write the feature names.
    label: bool
        Whether to write the label (y).
    """
    data = pd.DataFrame(dataset.X)

    if features:
        data.columns = dataset.features

    if label and dataset.y is not None:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index=False)
