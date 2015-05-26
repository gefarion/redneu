#!/usr/bin/python3
import nnpy

if __name__ == "__main__":

    momentums = [0, 0.3, 0.6, 0.9]
    hlayers_sizes = [[2], [4], [2, 2], [4, 4]]
    learning_rates = [0.1, 0.5]

    training_set = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]

    global_run = 1
    total_runs = len(momentums) * len(hlayers_sizes) * len(learning_rates) * 100

    validator = nnpy.NoneValidator(nnpy.IncrementalTrainer())
    fh = open('xor.txt', 'w')
    experiment = nnpy.Experiment(training_set, validator, 0.01, 10000, fh, 100)

    for hlayers_size in hlayers_sizes:
        for learning_rate in learning_rates:
            for momentum in momentums:

                def callback(run, ok, error, loops, partial_error):
                    global global_run
                    print("run #{} %{}".format(global_run, 100.0 * global_run / total_runs))
                    global_run += 1

                experiment.run(hlayers_size, learning_rate, momentum, callback)
