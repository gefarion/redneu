from redneu.hebb import HebbNeuralNetwork
import numpy as np
from redneu.utils import BOWDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dataset = BOWDataset(filename='tp2/tp2_training_dataset.csv')
reduced_dataset = dataset.group_words_by_cat(1)

print("dataset WORDS: {}".format(dataset.words_count()))

print("reduced dataset WORDS: {}".format(reduced_dataset.words_count()))

udataset = reduced_dataset.uncategorized_dataset()

def printe(hnn, epoch):
    print(epoch)

hnn = HebbNeuralNetwork(len(udataset[0]), 3)
hnn.train(udataset[:600], 50, ecallback=printe)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

markers = [u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

fout = open('reduced_50e.csv', 'w');

for data in reduced_dataset.dataset[600:]:
    y = hnn.activate(data[1:])
    fout.write("{},{},{},{}\n".format(data[0], y[0], y[1], y[2]))
    # ax.scatter([y[0]], [y[1]], [y[2]], marker=markers[data[0] - 1], c=colors[data[0] - 1])

# plt.show()