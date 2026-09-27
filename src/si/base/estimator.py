from abc import ABC, abstractmethod

from si.data.dataset import Dataset


class Estimator(ABC):
    """
    Classe base para qualquer objeto que "aprenda" a partir de dados.
    """

    def __init__(self, **kwargs):
        self.is_fitted = False

    @abstractmethod
    def _fit(self, dataset: Dataset):
        """
        Método abstrato responsável por estimar os parâmetros a partir
        dos dados. Deve ser implementado por todas as classes que
        estendem o Estimator.
        """
        raise NotImplementedError

    def fit(self, dataset: Dataset):
        """
        Ajusta o estimador ao dataset fornecido (chama _fit).

        Returns
        -------
        self: Estimator
        """
        self._fit(dataset)
        self.is_fitted = True
        return self
