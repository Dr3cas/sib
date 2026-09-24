import pandas as pd
import numpy as np

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ",", features: bool = True,
             label: bool = True) -> Dataset:
    """
    Lê um ficheiro csv e devolve um objeto Dataset.

    Parameters
    ----------
    filename: str
        Nome/caminho do ficheiro.
    sep: str
        Separador de valores.
    features: bool
        Se o ficheiro tem nomes de features (cabeçalho).
    label: bool
        Se o ficheiro tem y (assume-se que é a última coluna).

    Returns
    -------
    Dataset
    """
    df = pd.read_csv(filename, sep=sep, header=0 if features else None)

    if features:
        col_names = list(df.columns)
    else:
        col_names = None

    if label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        feature_names = col_names[:-1] if col_names else None
        label_name = col_names[-1] if col_names else None
    else:
        X = df.to_numpy()
        y = None
        feature_names = col_names
        label_name = None

    return Dataset(X=X, y=y, features=feature_names, label=label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ",",
              features: bool = True, label: bool = True) -> None:
    """
    Escreve um objeto Dataset para um ficheiro csv.

    Parameters
    ----------
    filename: str
        Nome/caminho do ficheiro.
    dataset: Dataset
        Objeto Dataset a escrever.
    sep: str
        Separador de valores.
    features: bool
        Se o ficheiro deve conter os nomes das features.
    label: bool
        Se o ficheiro deve conter y.
    """
    columns = list(dataset.features) if features else None
    df = pd.DataFrame(dataset.X, columns=columns)

    if label and dataset.y is not None:
        label_name = dataset.label if dataset.label else "y"
        df[label_name] = dataset.y

    df.to_csv(filename, sep=sep, index=False, header=features)
