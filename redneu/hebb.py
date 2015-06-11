#!/user/bin/python3
import numpy as np

class HebbNeuralNetwork():

    def __init__(self, ninputs, noutputs, learning_rate=None, use_oja=None):

        self.learning_rate = learning_rate
        self.use_oja = use_oja

        self.ninputs = ninputs
        self.noutputs = noutputs
        self.weights = np.random.uniform(-0.5, 0.5, (ninputs, noutputs))

    def ninputs(self):
        return self.ninputs

    def noutputs(self):
        return self.noutputs

    def weights(self):
        return self.weights

    def activate(self, input):

        x = np.array([input])
        y = np.dot(x, self.weights)
        return y[0]

    def train(self, dataset, epochs, ecallback=None):

        for e in range(1, epochs + 1):

            if ecallback: ecallback(self, e)
            learning_rate = self.learning_rate or (0.5 / e)

            for x in dataset:
                y = self.activate(x)
                dw = np.zeros((self.ninputs, self.noutputs), dtype=float)

                for j in range(0, self.noutputs):
                    for i in range(0, self.ninputs):
                        xe = 0
                        for k in range(0, self.use_oja and self.noutputs or (j + 1)):
                            xe += y[k] * self.weights[i][k]

                        dw[i][j] = learning_rate * y[j] * (x[i] - xe)

                self.weights += dw

class GHANeuralNetwork():

    def __init__(self, ninputs, noutputs, sigma0, alfa):

        self.sigma0 = sigma0
        self.alfa = alfa

        self.ninputs = ninputs
        self.noutputs = noutputs
        self.weights = np.random.uniform(-0.5, 0.5, (noutputs, ninputs))

    def ninputs(self):
        return self.ninputs

    def noutputs(self):
        return self.noutputs

    def weights(self):
        return self.weights

    def activate(self, input):

        x = np.array([input]).T
        y = np.dot(self.weights, x)
        return y.T[0]

    def train(self, dataset, epochs, callback=None):

        for t in range(1, epochs + 1):

            sigma = self.sigma0 * (t ** -self.alfa)
            tdw = 0

            for x in dataset:

                x = np.array([x]).T
                y = np.dot(self.weights, x)

                dw = sigma * ( np.dot(y, x.T) - np.dot(np.tril(np.dot(y, y.T)), self.weights) )
                self.weights += dw
                tdw += dw ** 2

            if callback: callback(self, t, tdw.sum() / len(dataset))

