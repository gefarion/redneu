from redneu.hebb import HebbNeuralNetwork, GHANeuralNetwork
import numpy as np
import sys

if __name__ == "__main__":

    A = (1, 5, 10, 15, 25, 50)
    EPOCHS = 1000

    dataset = []
    for i in range(0, 1000):
        point = [ np.random.uniform(-a, a) for a in A ]
        dataset.append(point)

    def call(hnn, t, tdw):
        print("\rTRAINING EPOCH: {} ({}%) tdw: {}".format(t, 100 * t / EPOCHS, tdw), end='')
        sys.stdout.flush()

    hnn = GHANeuralNetwork(6, 4, 0.0001, 0.5)
    hnn.train(dataset, EPOCHS, ecallback=call)

    outputs = [ hnn.activate(x) for x in dataset ]

    print("\nmean: {}".format(np.mean(outputs, axis=0)))

    print("std: {}".format(np.std(outputs, axis=0)))

    print("var: {}\n".format(np.var(outputs, axis=0)))

    for x in dataset[:3]:
        print("{} {}".format(x, hnn.activate(x)))

