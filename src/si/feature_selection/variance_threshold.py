import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    Seletor de features que remove todas as features cuja variância
    não ultrapassa um determinado limiar (threshold).
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        """
        Parameters
        ----------
        threshold: float
            Valor de corte (cut-off) da variância.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset):
        """Estima a variância de cada feature."""
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Seleciona todas as features com variância superior ao
        threshold.
        """
        mask = self.variance > self.threshold

        new_X = dataset.X[:, mask]
        new_features = np.array(dataset.features)[mask].tolist()

        return Dataset(X=new_X, y=dataset.y, features=new_features,
                        label=dataset.label)
