#!/usr/bin/python3
import nnpy
import network as network
import numpy as np
import matplotlib.pyplot as plt


def make_f(a):
    #return lambda x: np.sin(x) / 2 + 0.5
    return lambda x: 0.2 * np.sin(a * np.cos(x)) * np.cos(2 * x) + 0.25 * np.sin(x) + 0.5


def h(loops, error):
    print("#{} {}".format(loops, error))


if __name__ == "__main__":

    domain = np.arange(0 * np.pi, 2 * np.pi, 0.05)
    f = make_f(8)
    img = f(domain)

    training_set = []
    for x in domain:
        training_set.append((np.array([[x]]), np.array([[f(x)]])))


    nn = network.Network([1,32,1]);
    nn.SGD(training_set, 1000, 100, 0.2)

    nn_img = [nn.feedforward([x])[0][0] for x in domain]

    plt.plot(domain, img)
    plt.plot(domain, nn_img)
    plt.show()
