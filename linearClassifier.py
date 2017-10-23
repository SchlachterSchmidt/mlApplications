import numpy as np
from costFunction import costFunction
from drawUtil import *

def getData():
	x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
	y = np.sin(x) + 0.1*np.power(x,2) + 0.5*np.random.randn(100,1)
	x /= np.max(x)
	x = np.hstack((np.ones_like(x), x))
	order = np.random.permutation(len(x))
	portion = 20
	test_x = x[order[:portion]]
	test_y = y[order[:portion]]
	train_x = x[order[portion:]]
	train_y = y[order[portion:]]

	return x, y, train_x, train_y, test_x, test_y

def gradientDescent(w, x, y, tolerance):
	iterations = 1

	while True:
		gradient, error = costFunction(w, x, y)
		new_w = w - alpha * gradient

		if np.sum(abs(new_w - w)) < tolerance:
			print("Converged.")
			break

		# Print error every 50 iterations
		if iterations % 50 == 0:
			drawLine(fig, ax, x, x.dot(w), 'yellow', 'estimate')
			print("Iteration: %d: - Error: %.8f\nwith w: " % (np.int(iterations), error))
			print(w)

		iterations += 1
		w = new_w

	print("final cost: %8f" % costFunction(w, train_x, train_y)[1])

	return w, error


if __name__ == "__main__":
	alpha = 0.5
	tolerance = 1e-7

	x, y, train_x, train_y, test_x, test_y = getData()
	w = np.random.randn(2)

	fig, ax = getFigure()
	drawScatter(fig, ax, train_x, train_y, 'lightblue', 'training set')
	drawLine(fig, ax, x, x.dot(w), 'green', 'initial guess')

	w, error = gradientDescent(w, train_x, train_y, tolerance)

	drawLine(fig, ax, x, x.dot(w), 'red', 'decision boundary')
	drawScatter(fig, ax, test_x, test_y, 'purple', 'test set')

	print("cost on test set: %f" % costFunction(w, test_x, test_y)[1])
	input("Press Enter to close...")
