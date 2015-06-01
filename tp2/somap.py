from redneu.somap import SelfOrganizedMap
import numpy as np
from redneu.utils import BOWDataset
import matplotlib.pyplot as plt


def show_activation_layer(somap, dataset, category):

    output_size = somap.output_size
    activation_layer = np.zeros((output_size[0], output_size[1]))

    for data in dataset:
        if data[0] == category:
            y = somap.activate(data[1:])
            neuron = np.unravel_index(y.argmax(), y.shape)
            activation_layer[neuron[0]][neuron[1]] = 1

    plt.matshow(activation_layer)

dataset = BOWDataset(filename='tp2/tp2_training_dataset.csv')
reduced_dataset = dataset.group_words_by_cat(1)

print("dataset WORDS: {}".format(dataset.words_count()))

print("reduced dataset WORDS: {}".format(reduced_dataset.words_count()))

output_size = (40, 40)
udataset = reduced_dataset.uncategorized_dataset()

somap = SelfOrganizedMap(len(udataset[0]), output_size, 0.5, 0.5)

def printe(somap, epochs, lr, sg):
    print(epochs)

somap.train(udataset, 50, printe)

for cat in range(1, 10):
    show_activation_layer(somap, reduced_dataset.dataset, cat)

plt.show()