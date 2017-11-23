"""Module contains sigmoid function."""
import numpy as np


def sigmoid(z):
	"""Sigmoid function."""
	g = 1 / (1 + np.exp(-z))
	return g
