import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


N = 50
D = 50

# uniformly distributed numbers between -5, +5
X = (np.random.random((N, D)) - 0.5) * 10

# true weights - only the first 3 dimensions of X affect Y
true_w = np.array([1, 0.5, -0.5] + [0] * (D - 3))

# generate Y - add noise with variance 0.5
Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N) * 0.5))

# perform gradient descent to find w
costs = []  # keep track of all the costs
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 3.0

for t in range(5000):
	# update w
	Yhat = sigmoid(X.dot(w))
	delta = Yhat - Y
	w -= learning_rate * (X.T.dot(delta) + l1 * np.sign(w))

	# find and store the cost
	cost = -(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat)).mean() + l1 * np.abs(w).mean()
	costs.append(cost)

# plot the costs
plt.plot(costs)
plt.show()

# plot our w vs true w
plt.plot(true_w, label='true_w')
plt.plot(w, label='w map')
plt.legend()
plt.show()
