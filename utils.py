from systemtools.system import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
import multiprocessing

def fillTriangleMatrix(m, diagNumber=1.0):
	"""
		Convert this :

			np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
					  [0.5, np.nan, np.nan, np.nan, np.nan],
					  [0.9, 0.8, np.nan, np.nan, np.nan],
					  [0.3, 0.5, 0.6, np.nan, np.nan],
					  [0.2, 0.1, 0.9, 0.8, np.nan]])

		to this:

			array([[1. , 0.5, 0.9, 0.3, 0.2],
				   [0.5, 1. , 0.8, 0.5, 0.1],
				   [0.9, 0.8, 1. , 0.6, 0.9],
				   [0.3, 0.5, 0.6, 1. , 0.8],
				   [0.2, 0.1, 0.9, 0.8, 1. ]])

	"""
	m = np.nan_to_num(m)
	np.fill_diagonal(m, diagNumber)
	upper = np.transpose(m).copy()
	np.fill_diagonal(upper, 0.0)
	upper = np.triu(upper)
	return np.add(m, upper)

def pairwiseCosineSimilarity(data):
	"""
		This function compute the pairwise cosine similary of a vector.
		n is the dimension of the input vector, n*n is the size of the output matrix.
	"""
	return 1 - pairwise_distances(data, metric='cosine', n_jobs=multiprocessing.cpu_count())


def test1():
	pass


if __name__ == '__main__':
	test1()