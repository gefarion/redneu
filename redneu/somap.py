#!/user/bin/python3
import numpy as np

class SOMapClassifier():

    def _init_classmap(self, dataset):

        output_size = self.somap.output_size

        class_counter = np.zeros((output_size[0], output_size[1], max(dataset.categories()) + 1))
        for data in dataset.dataset:
            y = self.somap.activate(data[1:])
            neuron = np.unravel_index(y.argmax(), y.shape)
            class_counter[neuron[0]][neuron[1]][data[0]] += 1

        for i in range(output_size[0]):
         for j in range(output_size[1]):
                cmax = class_counter[i][j].argmax()
                if class_counter[i][j][cmax] > 0:
                    self.classmap[i][j] = cmax


    def __init__(self, somap, dataset):

        self.somap = somap
        self.classmap = np.full(somap.output_size, -1)
        self._init_classmap(dataset)


    def classify(self, data):
        y = self.somap.activate(data)
        neuron = np.unravel_index(y.argmax(), y.shape)
        return self.classmap[neuron[0]][neuron[1]]

    def fill_classmap(self):
        classmap = self.classmap
        for i in range(len(classmap)):
            for j in range(len(classmap[0])):
                if (classmap[i][j] < 0):
                    values = [
                        classmap[i-1][j],
                        classmap[(i+1) % len(classmap)][j],
                        classmap[i][j-1],
                        classmap[i][(j+1) % len(classmap[0])]
                    ]
                    classmap[i][j] = max(set(values), key=values.count)



class SelfOrganizedMap():

    def __init__(self, ninputs, output_size, learning_rate, sigma):

        self.learning_rate = learning_rate
        self.sigma = sigma

        self.ninputs = ninputs
        self.output_size = output_size
        self.noutputs = output_size[0] * output_size[1]

        self.neigx = np.arange(output_size[0])
        self.neigy = np.arange(output_size[1])

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

    def gaussian(self, c, sigma):

        d = 2 * np.pi * sigma * sigma
        ax = np.exp(-np.power(self.neigx - c[0], 2) / d)
        ay = np.exp(-np.power(self.neigy - c[1], 2)/ d)
        return np.outer(ax, ay)

    def proxy(self, p, sigma): # Funcion recomendada por la catedra (lenta)

        d = np.zeros(self.output_size)

        for i in range(0, self.output_size[0]):
            for j in range(0, self.output_size[1]):
                d[i][j] = np.e ** (- ((i - p[0])**2 + (j - p[1])**2) / (2 * (sigma ** 2)))

        return d

    def correction(self, input, learning_rate, sigma):

        y = self.activate(input)
        p = np.unravel_index(y.argmax(), y.shape)
        d = self.gaussian(p, sigma)
        dw = learning_rate * (np.array([input]).transpose() - self.weights) * d.flatten()
        self.weights += dw

        return dw

    def train(self, dataset, epochs, callback=None):

        for t in range(1, epochs + 1):
            eta = t ** (- self.learning_rate)
            sigma = t ** (- self.sigma)

            tdw = np.zeros((self.ninputs, self.noutputs))
            for x in dataset:
                tdw += self.correction(x, eta, sigma) ** 2

            if callback: callback(self, t, tdw.sum() / len(dataset), eta, sigma)
