from collections import Counter

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distância euclidiana entre um vetor x e cada linha de y."""
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


class KNNClassifier(Model):
    """
    Algoritmo dos k-vizinhos mais próximos para problemas de
    classificação. Estima a classe de uma amostra com base nas k
    amostras mais semelhantes do dataset de treino.
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

    def _get_closest_label(self, sample: np.ndarray):
        """Obtém a classe mais comum entre os k vizinhos mais próximos."""
        distances = self.distance(sample, self.dataset.X)
        k_nearest_idxs = np.argsort(distances)[: self.k]
        k_nearest_labels = self.dataset.y[k_nearest_idxs]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """Estima a classe para cada amostra do dataset de teste."""
        # Nota: não usamos np.apply_along_axis porque, quando as classes
        # são strings, o numpy infere o dtype (comprimento fixo) a
        # partir do primeiro resultado e trunca os seguintes.
        predictions = [self._get_closest_label(sample) for sample in dataset.X]
        return np.array(predictions)

    def _score(self, dataset: Dataset) -> float:
        """Calcula a accuracy entre as classes previstas e as reais."""
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)
