from abc import abstractmethod

from si.base.estimator import Estimator
from si.data.dataset import Dataset


class Model(Estimator):
    """
    Classe base para modelos de Machine Learning (classificadores e
    regressores). Um modelo é uma função matemática que, com base nas
    features, faz previsões para amostras de um dataset.
    """

    @abstractmethod
    def _predict(self, dataset: Dataset):
        """
        Método abstrato responsável por prever valores/classes para
        novas amostras. Deve ser implementado por todas as classes que
        estendem o Model.
        """
        raise NotImplementedError

    @abstractmethod
    def _score(self, dataset: Dataset) -> float:
        """
        Método abstrato responsável por calcular a métrica de erro do
        modelo, com um Dataset como input.
        """
        raise NotImplementedError

    def predict(self, dataset: Dataset):
        """
        Prevê valores/classes para o dataset fornecido (chama
        _predict). Verifica primeiro se o modelo já foi ajustado.
        """
        if not self.is_fitted:
            raise ValueError("O modelo ainda não foi ajustado (fit).")
        return self._predict(dataset)

    def score(self, dataset: Dataset) -> float:
        """
        Calcula a métrica de erro do modelo para o dataset fornecido.
        Verifica se o modelo está ajustado e, se sim, chama _score.
        """
        if not self.is_fitted:
            raise ValueError("O modelo ainda não foi ajustado (fit).")
        return self._score(dataset)
