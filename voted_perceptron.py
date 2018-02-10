import numpy as np

class voted_Perceptron(object):
    def __init__(self, eta=0.25, epochs=200):
        self.eta = eta
        self.epochs = epochs

    def training(self, X, Y):
        self.k = 0
        self.W = [np.zeros(len(X[0]))]
        self.c = [0]
        self.w = np.zeros(len(X[0]))
        i = 0
        while True:
            n_err = 0
            for xi, yi in zip(X, Y):
                if yi * (1 if np.dot(xi, self.w) >= 0 else -1) >= 0:
                    self.c[self.k] += 1
                else:
                    self.W.append([(w + yi*x) for w, x in zip(self.w, xi)])
                    self.w = self.W[self.k + 1]
                    self.c.append(1)
                    self.k += 1
                    n_err += 1

            i += 1
            if n_err == 0 or i > self.epochs:
                break
        if i > self.epochs:
            print "There is some error, but anyway the Hiperplane is: "
        return self.w, i-1, len(self.W)

    def net_input(self, xi):
        s = [(1 if np.dot(xi, self.W[i]) >= 0 else -1)*self.c[i] for i in np.arange(0, self.k+1, 1)]
        return sum(s)

    def predict(self, xi):
        return 1 if self.net_input(xi) >= 0 else -1
