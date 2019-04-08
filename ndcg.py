import pandas as pd
from newssource.metrics.rank_metrics import ndcg_at_k
import numpy as np

def pairwiseSimNDCG(simMatrix, labels, logger=None, verbose=True):
    """
        This function take a similary matrix n*n (which is a symmetric matrix)
        with 1.0 on the diagonal.
        It take labels which are class identifiers of each dimension.
        Can be strings, for example the author of an article:
        ["author1", "author1", "author2", "author3", "author3", "author3"]
        It returns the nDCG at k (with k = n) averaged over all columns.
    """
    labels = pd.factorize(labels)[0]
    def rankLabels(col):
        # col = np.array([row[0] + row[1], row[2] + row[3]])
        col = np.vstack([col, labels])
        # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
        # argsort() to get indexes and sort, [::-1] for reverse
        col = col[:, col[0,:].argsort()[::-1]]
        col = col[1]
        return col
    simMatrix = np.apply_along_axis(rankLabels, 0, simMatrix)
    nDCGs = []
    for x in range(simMatrix.shape[0]):
        col = simMatrix[:, x]
        label = labels[x]
        for y in range(len(col)):
            col[y] = col[y] == label
        nDCGs.append(ndcg(col))
    return np.average(nDCGs)



def ndcg(r, method=0):
    """
        This function return the nDCG at k with k = len(r)
    """
    return ndcg_at_k(r, len(r), method=method)