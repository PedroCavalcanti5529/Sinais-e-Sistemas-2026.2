"""Funções compartilhadas."""
import numpy as np
from scipy.stats import rankdata


def auc_rank(X, y):
    """AUC = P(positivo > negativo) por coluna, vetorizada (equivale a
    mannwhitneyu(pos, neg).statistic / (n_pos * n_neg)).
    X: (n, m); y: bool (n,) com True = classe positiva."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.full(X.shape[1], np.nan)
    r = rankdata(X, axis=0)
    return (r[y].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
