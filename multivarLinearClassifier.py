import numpy as np
import matplotlib.pyplot as plt
from costFunction import costFunction
from drawUtil import *

def getData(model_order):
	portion = 20
	x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
	y = np.sin(x) + 0.1 * np.power(x, 2) + 0.5 * np.random.randn(100, 1)
	x = np.power(x, range(model_order))
	x /= np.max(x, axis=0)
	order = np.random.permutation(len(x))
	test_x = x[order[:portion]]
	test_y = y[order[:portion]]
	train_x = x[order[portion:]]
	train_y = y[order[portion:]]

	return x, y, train_x, train_y, test_x, test_y

def stochasticGradientDescent(w, x, y, tolerance, batch_size, alpha, decay):
	epochs = 1
	iterations = 0
	while True:
		order = np.random.permutation(len(train_x))
		x = x[order]
		y = y[order]
		b=0
		while b < len(train_x):
			tx = x[b : b+batch_size]
			ty = y[b : b+batch_size]
			gradient = costFunction(w, tx, ty)[0]
			error = costFunction(w, x, y)[1]
			w -= alpha * gradient
			iterations += 1
			b += batch_size

		# Keep track of our performance
		if epochs%100==0:
			new_error = costFunction(w, x, y)[1]
			print("Epoch: %d - Error: %.4f" %(epochs, new_error))
			#drawLine(fig, ax, x, x.dot(w), 'yellow', 'estimate')
			# Stopping Condition
			if abs(new_error - error) < tolerance:
				print("Converged.")
				break

		alpha = alpha * (decay ** int(epochs/1000))
		epochs += 1

	return w, error, iterations

if __name__ == "__main__":
	model_order = 6
	alpha = 0.5
	tolerance = 1e-9
	decay = 0.99
	batch_size = 10

	x, y, train_x, train_y, test_x, test_y = getData(model_order)
	w = np.random.randn(model_order)

	fig, ax = getFigure()
	drawScatter(fig, ax, train_x, train_y, 'lightblue', 'training set')
	drawLine(fig, ax, x, x.dot(w), 'green', 'initial guess')

	w, error, iterations = stochasticGradientDescent(w, train_x, train_y, tolerance, batch_size, alpha, decay)

	drawLine(fig, ax, x, x.dot(w), 'red', 'decision boundary')
	drawScatter(fig, ax, test_x, test_y, 'purple', 'test set')

	print("w =",w)
	print("Test Cost = %.8f" % costFunction(w, test_x, test_y)[1])
	print("Total iterations =", iterations)
	input("Press Enter to close...")
