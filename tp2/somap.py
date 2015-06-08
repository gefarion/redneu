from redneu.somap import SelfOrganizedMap, SOMapClassifier
from redneu.hebb import GHANeuralNetwork
import numpy as np
from redneu.utils import BOWDataset
import matplotlib.pyplot as plt
import sys

def show_activation_layer(somap, dataset, category):

    output_size = somap.output_size
    activation_layer = np.zeros((output_size[0], output_size[1]))

    for data in dataset:
        if data[0] == category:
            y = somap.activate(data[1:])
            neuron = np.unravel_index(y.argmax(), y.shape)
            activation_layer[neuron[0]][neuron[1]] = 1

    plt.matshow(activation_layer)


print("Cargando dataset...")
dataset = BOWDataset(filename='tp2/tp2_training_dataset.csv')
# dataset = dataset.group_words_by_cat(1)

print("Entrenando GHA...")

def gha_call(hnn, t, tdw):
    print("\rTRAINING EPOCH: {} ({:.2%}) tdw: {}".format(t, t / GHA_EPOCHS, tdw), end='')
    sys.stdout.flush()

LIMIT = 700

GHA_EPOCHS = 200
ghann = GHANeuralNetwork(len(dataset.dataset[0]) - 1, 10, 0.0001, 0.1)
ghann.train(dataset.uncategorized_dataset()[:LIMIT], GHA_EPOCHS, callback=gha_call)

reduced_dataset = dataset.activate_neural_network(ghann)

print("\n\nEntrenando SOMAP...")

output_size = (20, 20)
udataset = reduced_dataset.uncategorized_dataset()

somap = SelfOrganizedMap(len(udataset[0]), output_size, 0.5, 0.3)

SOMAP_EPOCHS = 10000
def somap_call(somap, t, tdw, lr, sg):
    print("\rTRAINING EPOCH: {} ({:.2%}) tdw: {}".format(t, t / SOMAP_EPOCHS, tdw), end='')
    sys.stdout.flush()

somap.train(udataset[:LIMIT], SOMAP_EPOCHS, somap_call)

classifier = SOMapClassifier(somap, reduced_dataset.slice(None, LIMIT))

total = [0, 0, 0]
total_by_class = [ [0, 0, 0] for i in range(9) ]
for data in reduced_dataset.dataset[LIMIT:]:
    c = int(classifier.classify(data[1:]))
    if (c == -1):
        total[2] += 1
        total_by_class[data[0] - 1][2] += 1
    elif (c == data[0]):
        total[0] += 1
        total_by_class[data[0] - 1][0] += 1
    else:
        total[1] += 1
        total_by_class[data[0] - 1][1] += 1

def stats(total):
    return "hits: {} ({:.2%}), miss: {} ({:.2%}), desc: {} ({:.2%})".format(total[0], total[0]/sum(total), total[1], total[1]/sum(total), total[2], total[2]/sum(total))

print("\n\nResultado: " + stats(total))

for c in range(9):
    print("[{}] Resultado: ".format(c) + stats(total_by_class[c]))


plt.matshow(classifier.classmap)
plt.show()

# for cat in range(1, 10):
#     show_activation_layer(somap, reduced_dataset.dataset, cat)

# plt.show()
print('')