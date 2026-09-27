from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> tuple:
    """
    Realiza uma análise de variância (ANOVA) unidirecional entre cada
    feature e a classe (y) do dataset.

    Parameters
    ----------
    dataset: Dataset
        Dataset a analisar.

    Returns
    -------
    tuple(tuple, tuple)
        (F values, p values) - um valor por feature.
    """
    classes = dataset.get_classes()

    groups = [dataset.X[dataset.y == c, :] for c in classes]

    F, p = stats.f_oneway(*groups)
    return F, p
