import numpy as np
import pandas as pd


class Dataset:
    """
    Representa um dataset de Machine Learning, guardando a matriz de
    features (X), o vetor da variável dependente (y), os nomes das
    features e o nome do label.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None,
                 features: list = None, label: str = None):
        """
        Parameters
        ----------
        X: np.ndarray
            Matriz/tabela de features (variáveis independentes).
        y: np.ndarray
            Vetor da variável dependente.
        features: list
            Vetor com os nomes das features.
        label: str
            Nome da variável dependente.
        """
        if X is None:
            raise ValueError("X não pode ser None")

        if features is None:
            features = [f"feat_{i}" for i in range(X.shape[1])]
        else:
            features = list(features)

        if y is not None and label is None:
            label = "y"

        self.X = X
        self.y = y
        self.features = features
        self.label = label

    @property
    def shape(self) -> tuple:
        """Devolve a dimensão do dataset (n_samples, n_features)."""
        return self.X.shape

    def has_label(self) -> bool:
        """Verifica se o dataset tem variável dependente (y)."""
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """Devolve as classes possíveis do dataset (valores únicos de y)."""
        if not self.has_label():
            raise ValueError("O dataset não tem y (label)")
        return np.unique(self.y)

    def get_mean(self) -> np.ndarray:
        """Devolve a média de cada feature."""
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """Devolve a variância de cada feature."""
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """Devolve a mediana de cada feature."""
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """Devolve o valor mínimo de cada feature."""
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """Devolve o valor máximo de cada feature."""
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """Devolve um pandas DataFrame com todas as métricas descritivas."""
        return pd.DataFrame(
            {
                "mean": self.get_mean(),
                "variance": self.get_variance(),
                "median": self.get_median(),
                "min": self.get_min(),
                "max": self.get_max(),
            },
            index=self.features,
        )

    # ---------------------------------------------------------------
    # Exercício 2: tratamento de valores nulos / remoção de amostras
    # ---------------------------------------------------------------

    def dropna(self):
        """
        Remove todas as amostras (linhas) que contenham pelo menos um
        valor nulo (NaN). Atualiza também o vetor y em conformidade.

        Returns
        -------
        self: Dataset
        """
        mask = ~np.isnan(self.X).any(axis=1)
        self.X = self.X[mask]
        if self.y is not None:
            self.y = self.y[mask]
        return self

    def fillna(self, value):
        """
        Substitui todos os valores nulos (NaN) por um valor fixo, pela
        média ou pela mediana de cada feature.

        Parameters
        ----------
        value: float | str
            Valor fixo a usar, ou "mean" / "median".

        Returns
        -------
        self: Dataset
        """
        if value == "mean":
            fill_values = self.get_mean()
        elif value == "median":
            fill_values = self.get_median()
        else:
            fill_values = np.full(self.X.shape[1], value)

        inds = np.where(np.isnan(self.X))
        self.X[inds] = np.take(fill_values, inds[1])
        return self

    def remove_by_index(self, index: int):
        """
        Remove uma amostra do dataset pelo seu índice. Atualiza também
        o vetor y em conformidade.

        Parameters
        ----------
        index: int
            Índice da amostra a remover.

        Returns
        -------
        self: Dataset
        """
        self.X = np.delete(self.X, index, axis=0)
        if self.y is not None:
            self.y = np.delete(self.y, index, axis=0)
        return self

    def __repr__(self):
        return f"Dataset(shape={self.shape}, label={self.label})"
