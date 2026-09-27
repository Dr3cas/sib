from abc import abstractmethod

from si.base.estimator import Estimator
from si.data.dataset import Dataset


class Transformer(Estimator):
    """
    Tipo específico de Estimator usado para modificar/transformar
    dados (ex.: seleção de features).
    """

    @abstractmethod
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Método abstrato responsável por transformar os dados. Deve ser
        implementado por todas as classes que estendem o Transformer.
        """
        raise NotImplementedError

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Aplica a transformação aprendida aos dados (chama _transform).
        """
        return self._transform(dataset)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Ajusta o transformer aos dados e de seguida transforma-os
        (chama fit e depois transform).
        """
        self.fit(dataset)
        return self.transform(dataset)
