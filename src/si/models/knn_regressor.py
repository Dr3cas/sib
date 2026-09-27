import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distância euclidiana entre um vetor x e cada linha de y."""
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


class KNNRegressor(Model):
    """
    Algoritmo dos k-vizinhos mais próximos para problemas de
    regressão. Estima o valor de uma amostra com base na média dos
    valores das k amostras mais semelhantes do dataset de treino.
    """

    def __init__(self, k: int = 5, distance=euclidean_distance, **kwargs):
        """
        Parameters
        ----------
        k: int
            Número de exemplos vizinhos a considerar.
        distance: callable
            Função que calcula a distância entre uma amostra e as
            amostras do dataset de treino.
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset):
        """Guarda o dataset de treino."""
        self.dataset = dataset
        return self

    def _get_closest_value(self, sample: np.ndarray) -> float:
        """Calcula a média dos valores dos k vizinhos mais próximos."""
        distances = self.distance(sample, self.dataset.X)
        k_nearest_idxs = np.argsort(distances)[: self.k]
        k_nearest_values = self.dataset.y[k_nearest_idxs]
        return np.mean(k_nearest_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """Estima o valor para cada amostra do dataset de teste."""
        return np.apply_along_axis(self._get_closest_value, axis=1,
                                    arr=dataset.X)

    def _score(self, dataset: Dataset) -> float:
        """Calcula o rmse entre os valores previstos e os reais."""
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)
