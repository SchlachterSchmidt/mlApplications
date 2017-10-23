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
