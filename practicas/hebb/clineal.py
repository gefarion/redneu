from redneu.hebb import HebbNeuralNetwork
import numpy as np


if __name__ == "__main__":

    M1 = [[4, 0, 0.5], [0, 2, 0.5]]
    M2 = [[1, 0.5, 0.75], [0.5, 1, 0.75]]
    M3 = [[1, 1, 1], [0.5, 0.5, 0.5]]

    M = np.array(M1)

    dataset = []
    for i in range(0, 1000):
        dataset.append(np.dot(np.random.uniform(-1, 1, (1, 2)), M)[0])

    hnn = HebbNeuralNetwork(3, 2, learning_rate=0.001)
    hnn.train(dataset, 100)

    print("M: {}\n".format(M))
    print("W: {}\n".format(hnn.weights))
    print("Wt: {}\n".format(hnn.weights.transpose()))

    outputs = [ hnn.activate(x) for x in dataset ]

    print("mean: {}".format(np.mean(outputs, axis=0)))

    print("std: {}".format(np.std(outputs, axis=0)))

    print("var: {}\n".format(np.var(outputs, axis=0)))

    for x in dataset[:10]:
        y = hnn.activate(x)
        xe = np.dot(np.array([y]), hnn.weights.transpose())[0]
        print("{}{}{}".format(x, y, xe))



