from redneu.hebb import GHANeuralNetwork
import numpy as np
from redneu.utils import BOWDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

dataset = BOWDataset(filename='tp2/tp2_training_dataset.csv')
tdataset = dataset.uncategorized_dataset()

EPOCHS = 100
def call(hnn, t, tdw):
    print("\rTRAINING EPOCH: {} ({}%) tdw: {}".format(t, 100 * t / EPOCHS, tdw), end='')
    sys.stdout.flush()

hnn = GHANeuralNetwork(len(tdataset[0]), 3, 0.0001, 0.1)
hnn.train(tdataset[:600], EPOCHS, callback=call)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

markers = [u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

# fout = open('reduced_50e.csv', 'w');

reduced_dataset = dataset.activate_neural_network(hnn)

for data in reduced_dataset.dataset[600:]:
    ax.scatter([data[1]], [data[2]], [data[3]], marker=markers[data[0] - 1], c=colors[data[0] - 1])
    pass

plt.show()
print();