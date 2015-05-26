#!/usr/bin/python3
import nnpy
# import numpy as np
import gnumpy as np
import matplotlib.pyplot as plt


def make_f(a):
    # return lambda x: np.sin(x) / 2 + 0.5
    return lambda x: 0.2 * np.sin(a * np.cos(x)) * np.cos(2 * x) + 0.25 * np.sin(x) + 0.5


def h(loops, error):
    print("#{} {}".format(loops, error))


if __name__ == "__main__":

    domain = np.arange(0 * np.pi, 2 * np.pi, 0.05)
    f = make_f(8)
    img = f(domain)

    training_set = []
    for x in domain:
        training_set.append(([x], [f(x)]))

    nn = nnpy.MultiLayerPerceptron(1, 1, [256, 256], 0.1, 0.6)
    validator = nnpy.NoneValidator(nnpy.MiniBatchTrainer(10))
    (ok, error, loops, partial_errors) = validator.validate(nn, training_set, 0.001, 500000, h)

    nn_img = [nn.activate([x])[0] for x in domain]

    plt.plot(domain, img)
    plt.plot(domain, nn_img)
    plt.show()



