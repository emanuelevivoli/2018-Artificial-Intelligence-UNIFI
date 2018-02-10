import numpy as np
import math

class Perceptron(object):
    def __init__(self, eta=0.25, epochs=200):
        self.eta = eta
        self.epochs = epochs

    def training(self, X, Y):
        self.w = np.zeros(len(X[0]))
        k = 0
        i = 0
        max_ = self.max_norm(X)
        while True:
            n_err = 0
            for xi, yi in zip(X, Y):
                if yi*self.predict(xi) < 0:
                    self.w[1:] = [x + self.eta*(y*yi) for x, y in zip(self.w[1:], xi)]
                    self.w[0] += self.eta * yi * math.pow(max_, 2)
                    k += 1
                    n_err += 1
            i += 1
            if n_err == 0 or i > self.epochs:
                break
        if i > self.epochs:
            print "There is some error, but anyway the Hiperplane is: "
        return self.w, i-1, k

    def net_input(self, xi):
        return np.dot(xi, self.w)

    def predict(self, xi):
        return 1 if self.net_input(xi) >= 0.0 else -1

    def max_norm(self, X):
        v = np.linalg.norm(X[0])
        for i in range(1, len(X)):
            if np.linalg.norm(X[i]) > v:
                v = np.linalg.norm(X[i])
        return v