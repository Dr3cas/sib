import numpy as np

from si.data.dataset import Dataset


def read_data_file(filename: str, sep: str = ",", label: bool = True) -> Dataset:
    """
    Lê um ficheiro de dados genérico (sem cabeçalho) usando numpy e
    devolve um objeto Dataset.

    Parameters
    ----------
    filename: str
        Nome/caminho do ficheiro.
    sep: str
        Separador de valores.
    label: bool
        Se o ficheiro tem y (assume-se que é a última coluna).

    Returns
    -------
    Dataset
    """
    raw = np.genfromtxt(filename, delimiter=sep)

    if label:
        X = raw[:, :-1]
        y = raw[:, -1]
    else:
        X = raw
        y = None

    return Dataset(X=X, y=y)


def write_data_file(filename: str, dataset: Dataset, sep: str = ",",
                     label: bool = True) -> None:
    """
    Escreve um objeto Dataset para um ficheiro de dados genérico usando
    numpy.

    Parameters
    ----------
    filename: str
        Nome/caminho do ficheiro.
    dataset: Dataset
        Objeto Dataset a escrever.
    sep: str
        Separador de valores.
    label: bool
        Se deve escrever y (última coluna).
    """
    if label and dataset.y is not None:
        data = np.column_stack((dataset.X, dataset.y))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)
