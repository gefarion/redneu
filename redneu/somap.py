#!/user/bin/python3
import numpy as np

class SelfOrganizedMap():

    def __init__(self, ninputs, output_size, learning_rate_coef, sigma_coef):

        self.learning_rate_coef = learning_rate_coef
        self.sigma_coef = sigma_coef

        self.ninputs = ninputs
        self.output_size = output_size
        self.noutputs = output_size[0] * output_size[1]

        self.weights = np.random.uniform(-0.5, 0.5, (self.ninputs, self.noutputs))

    def inputs(self):
        return self.ninputs

    def noutputs(self):
        return self.noutputs

    def output_size(self):
        return self.output_size

    def weights(self):
        return self.weights

    def activate(self, input):

        y = np.linalg.norm(self.weights - np.array([input]).transpose(), None, 0)
        r = np.zeros_like(y)
        r[y.argmin()] = 1

        return r.reshape(self.output_size);

    def _proxy(self, p, sigma):

        d = np.zeros(self.output_size)
        for i in range(0, self.output_size[0]):
            for j in range(0, self.output_size[1]):
                d[i][j] = np.e ** (- ((i - p[0])**2 + (j - p[1])**2) / (2 * (sigma ** 2)))

        return d

    def _correction(self, input, learning_rate, sigma):

        y = self.activate(input)
        p = np.unravel_index(y.argmax(), y.shape)
        d = self._proxy(p, sigma)
        dw = learning_rate * (np.array([input]).transpose() - self.weights) * d.flatten()
        self.weights += dw

    def train(self, dataset, epochs, ecallback=None):

        for t in range(1, epochs + 1):
            learning_rate = t ** (- self.learning_rate_coef)
            sigma = t ** (- self.sigma_coef)

            if ecallback: ecallback(self, t, learning_rate, sigma)

            for x in dataset:
                self._correction(x, learning_rate, sigma)
