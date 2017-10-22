import numpy as np

def costFunction(w, x, y):
	y_estimate = x.dot(w).flatten()
	error = (y.flatten() - y_estimate)
	mse = (1.0 / len(x)) * np.sum(np.power(error, 2))
	gradient = -(1.0/len(x)) * error.dot(x)

	return gradient, mse
