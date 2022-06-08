import numpy as np


class logisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr  # learning rate, default value that is usually very small
        self.n_iters = n_iters  # number of iterations for gradient descent
        self.weights = None  # come up with these two
        self.bias = None
        self.losses = []

    # takes in training samples and values/labels
    # parameters:
    # X = numpy nd vector of (n(number of samples) * m(number of features for each sample))
    # y = id row vector, size m
    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape  # unpack the shape of the data
        self.weights = np.zeros(n_features)  # parameters
        self.bias = 0  # parameters

        # gradient descent
        for _ in range(self.n_iters):
            # we will use the sigmoid/logistic function to map input values from a wide range into a
            # limited interval
            # The logistic function will be our hypothesis function with range between 1 and 0
            linear_model = np.dot(X, self.weights) + self.bias  # (w * x) + b
            y_predicted = self._sigmoid(linear_model)  # y_hat

            # Applying update rules
            # dw is the partial derivative of the Loss function with respect to w
            # db is the partial derivative of the Loss function with respect to b

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Gradient of loss w.r.t weights.
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Gradient of loss w.r.t bias.

            # update our parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            l = self.loss(y, self._sigmoid(np.dot(X, self.weights) + self.bias))
            self.losses.append(l)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls, y_predicted

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, y, y_predicted):
        loss = -np.mean(y * (np.log(y_predicted)) - (1 - y) * np.log(1 - y_predicted))
        return loss

    def the_goods(self):
        print(f"weights = {self.weights}")
        print(f"bias = {self.bias}")
        print(f"losses = {self.losses}")
        return self.weights, self.bias, self.losses
