from redneu.somap import SelfOrganizedMap
import numpy as np
import matplotlib.pyplot as plt

def generar_puntos(limits, points_per_region):

    regions_points = [[] for x in limits]

    for r in range(len(limits)):
        # if r == 3: points_per_region *= 4

        regions_points[r] = list(zip(
            np.random.uniform(limits[r][0][0], limits[r][0][1], points_per_region),
            np.random.uniform(limits[r][1][0], limits[r][1][1], points_per_region)
        ))

    return regions_points

if __name__ == "__main__":

    # Generacion de puntos
    limits = [
        [(1,2),(1,2)],
        [(4,5),(1,2)],
        [(1,2),(4,5)],
        [(4,5),(4,5)],
    ]

    regions_points = generar_puntos(limits, 100)

    all_points = []
    for points in regions_points:
        all_points += points

    # Dibujo de los puntos
    # for point in all_points:
    #     plt.plot(point[0],point[1], 'b+')
    # plt.axis([0, 6, 0, 6])
    # plt.show()

    # Creo y entreno la red
    output_size = (10, 10)
    somap = SelfOrganizedMap(2, output_size, 0.5, 0.5)
    def printe(somap, epochs, , tdw, lr, sg):
        print(epochs)
    somap.train(all_points, 100, printe)

    # Generacion de la capa clasificadora
    output_region_count = np.zeros((output_size[0], output_size[1], 4))
    for r in range(4):
        for point in regions_points[r]:
            y = somap.activate(point)
            neuron = np.unravel_index(y.argmax(), y.shape)
            output_region_count[neuron[0]][neuron[1]][r] += 1

    class_layer = np.full(output_size, -1)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            imax = output_region_count[i][j].argmax()
            if output_region_count[i][j][imax] > 0:
                class_layer[i][j] = imax

    # Prueba del clasificador
    test_points = generar_puntos(limits, 500)

    total_hit = 0
    total_miss = 0
    for r in range(4):
        region_hit = 0
        region_miss = 0

        for point in test_points[r]:
            y = somap.activate(point)
            neuron = np.unravel_index(y.argmax(), y.shape)
            rclass = class_layer[neuron[0]][neuron[1]]

            if r == rclass:
                region_hit += 1
            else:
                region_miss += 1

        total_miss += region_miss
        total_hit += region_hit

        print ("REGION {}: hit {}%  miss {}%".format(r,
            100.0 * region_hit / (region_hit + region_miss),
            100.0 * region_miss / (region_hit + region_miss))
        )

    print ("TOTAL: hit {}%  miss {}%".format(
        100.0 * total_hit / (total_hit + total_miss),
        100.0 * total_miss / (total_hit + total_miss))
    )


    print(class_layer)
    plt.matshow(class_layer)
    plt.show()
