import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula a accuracy entre os valores reais e os valores previstos.
    """
    return np.sum(y_true == y_pred) / len(y_true)