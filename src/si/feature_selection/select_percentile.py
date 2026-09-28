from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Selects a percentage of the features with the highest F values,
    computed by a variance analysis function (score_func).
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 10, **kwargs):
        """
        Parameters
        ----------
        score_func: callable
            Variance analysis function (f_classification by default).
        percentile: float
            Percentile of features to select (e.g., 40 to keep 40% of the features).
        """
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimates the F and p values of each feature using the score_func.

        Returns
        -------
        self: SelectPercentile
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the features whose F value is above the threshold. Ties at the
        threshold are used (in the original order of the features) to reach the
        number of features that corresponds to the percentile.

        Example: F = [1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2] and percentile=40.
        The threshold is the 60th percentile (5.6). The mask keeps F > 5.6 -> [7.8, 6.5].
        We need 4 features (40% of 10), so the first two tied features ([5.6, 5.6]) are added.
        The selected features are [5.6, 5.6, 7.8, 6.5].
        """
        F = np.asarray(self.F)
        n_select = int(len(F) * self.percentile // 100)

        if n_select <= 0:
            return Dataset(dataset.X[:, :0], dataset.y, features=[], label=dataset.label)

        # F value at the (100 - percentile)-th percentile
        threshold = np.percentile(F, 100 - self.percentile)

        mask = F > threshold
        n_selected = int(mask.sum())

        if n_selected < n_select:
            # complete with the first tied features (F == threshold)
            ties = np.where(np.isclose(F, threshold))[0]
            mask[ties[:n_select - n_selected]] = True
        elif n_selected > n_select:
            # too many features: keep only the ones with the highest F
            selected = np.where(mask)[0]
            keep = selected[np.argsort(F[selected])[::-1][:n_select]]
            mask = np.zeros_like(mask)
            mask[keep] = True

        new_X = dataset.X[:, mask]
        new_features = np.array(dataset.features)[mask].tolist()
        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)
