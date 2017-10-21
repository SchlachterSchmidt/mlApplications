import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def getFigure():
	plt.ion()
	fig = plt.figure()
	ax = fig.add_axes([0, 0, 1, 1])
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	return fig, ax

def drawScatter(fig, ax, x, y, color, label):
	ax.scatter(x[:,1], y, color=color, linewidth=0.3, label=label)
	plt.show()
	plt.pause(0.5)

def drawLine(fig, ax, x, y, color, label):
	ax.plot(x[:,1], y, color=color, linewidth=0.5, label=label)
	plt.draw()
	plt.pause(0.5)

def costFunction(w, x, y):
	y_estimate = x.dot(w).flatten()
	error = (y.flatten() - y_estimate)
	mse = (1.0 / len(x)) * np.sum(np.power(error, 2))
	gradient = -(1.0/len(x)) * error.dot(x)

	return gradient, mse

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
