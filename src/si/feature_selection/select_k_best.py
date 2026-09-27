import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Seletor de features que seleciona as k features com maior score
    (F value), calculado por uma função de análise de variância.
    """

    def __init__(self, score_func=f_classification, k: int = 10, **kwargs):
        """
        Parameters
        ----------
        score_func: callable
            Função de análise de variância (f_classification por defeito).
        k: int
            Número de features a selecionar.
        """
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset):
        """Estima os valores F e p para cada feature."""
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """Seleciona as k features com maior valor F."""
        idxs = np.argsort(self.F)[-self.k:]
        idxs = np.sort(idxs)

        new_X = dataset.X[:, idxs]
        new_features = np.array(dataset.features)[idxs].tolist()

        return Dataset(X=new_X, y=dataset.y, features=new_features,
                        label=dataset.label)
