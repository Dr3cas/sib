import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Root Mean Squared Error (RMSE) entre os valores reais e
    os valores previstos.
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))