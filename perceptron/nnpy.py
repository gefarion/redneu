#!/user/bin/python3
import numpy as np


class Trainer(object):

    def train(self, neural_network, training_set):

        raise NotImplementedError('subclasses must override!')


class IncrementalTrainer(Trainer):

    def train(self, neural_network, training_set):

        abs_error = 0

        for i in np.random.permutation(len(training_set)):
            input, output = training_set[i]

            neural_network.activate(input)
            abs_error += neural_network.correction(output)
            neural_network.adaptation()

        return abs_error


class MiniBatchTrainer(Trainer):

    def __init__(self, batch_num):
        self.batch_num = batch_num

    def train(self, neural_network, training_set):

        count = 0
        sum_error = 0

        for i in np.random.permutation(len(training_set)):
            count += 1
            input, output = training_set[i]

            neural_network.activate(input)
            sum_error += neural_network.correction(output)

            if count % self.batch_num == 0:
                neural_network.adaptation()

        if count % self.batch_num != 0:
            neural_network.adaptation()

        # sum_error = 0
        # for input, output in training_set:
        #     sum_error += neural_network.error(input, output)

        return sum_error


class BatchTrainer(Trainer):

    def train(self, neural_network, training_set):

        abs_error = 0

        for i in np.random.permutation(len(training_set)):
            input, output = training_set[i]

            neural_network.activate(input)
            abs_error += neural_network.correction(output)

        neural_network.adaptation()

        return abs_error


class Validator(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def __test(self, neural_network, test_set):

        error = 0

        for test_case in test_set:
            input, output = test_case
            result = neural_network.activate(input)
            error += sum((output - result)[0] ** 2)

        return error

    def validate(self, neural_network, training_set, min_error, max_loops):

        raise NotImplementedError('subclasses must override!')


class NoneValidator(Validator):

    def validate(self, neural_network, training_set, min_error, max_loops, handler=None):

        error = 1
        loops = 0
        partial_errors = []

        while (error > min_error and loops < max_loops):
            error = self.trainer.train(neural_network, training_set)

            if (handler):
                handler(loops, error)

            # partial_errors.append(error)
            loops += 1

        if error <= min_error:
            return True, error, loops, partial_errors
        else:
            return False, error, loops, partial_errors


class CrossValidator(Validator):

    def __init__(self, trainer, fold_number=10):

        super.__init__(trainer)
        self.fold_number = fold_number

    def __generate_folding_sets(self, training_set):
        pass

    def validate(self, neural_network, training_set, min_error, max_loops):

        folding_sets = self.__generate_folding_sets(training_set)
        verrors = np.array([], dtype=float)
        traning_min_error = 100

        partial_errors = []

        for folding_set in folding_sets:

            titems, vitems = folding_set
            error = 1
            loops = 0

            while (error > min_error and loops < max_loops):

                error = self.trainer.train(neural_network, titems)
                partial_errors.append(error)
                loops += 1

            verrors.append(self.__test(neural_network, vitems))
            traning_min_error = min(traning_min_error, error)

        verror = np.average(verrors)

        if traning_min_error <= min_error:
            return True, verror, loops, partial_errors
        else:
            return False, verror, loops, partial_errors


class HoldOutValidator(Validator):

    def __init__(self, trainer, split_rate):

        super.__init__(trainer)
        self.split_rate = split_rate

    def __split_training_set(self, training_set):

        pitems = np.random.permutation(len(training_set))
        split_index = int(len(pitems) * self.split_rate)
        titems = pitems[:split_index]
        vitems = pitems[split_index:]

        return titems, vitems

    def validate(self, neural_network, training_set, min_error, max_loops):

        titems, vitems = self.__split_training_set(training_set, training_set)
        error = 1
        loops = 0
        partial_errors = []

        while (error > min_error and loops < max_loops):

            error = self.trainer.train(neural_network, titems)
            partial_errors.append(error)
            loops += 1

        verror = self.__test(neural_network, vitems)

        if error <= min_error:
            return True, verror, loops, partial_errors
        else:
            return False, verror, loops, partial_errors


class NeuralNetwork(object):

    def activate(self, input):

        raise NotImplementedError('subclasses must override!')

    def correction(self, output):

        raise NotImplementedError('subclasses must override!')

    def adaptation(self):

        raise NotImplementedError('subclasses must override!')


class MultiLayerPerceptron(NeuralNetwork):

    def __sigmoid(self, x):

        return 1 / (1 + np.e ** ((-self.beta) * x))

    def __init_weights(self):

        for i in range(1, self.nlayers):
            self.weights.append(
                np.random.uniform(-0.5, 0.5,
                    (self.layers_sizes[i - 1], self.layers_sizes[i])
                )
            )

    def __init_delta_weights(self):

        for i in range(1, self.nlayers):
            self.delta_weights.append(
                np.zeros(
                    (self.layers_sizes[i - 1], self.layers_sizes[i]),
                    dtype=float
                )
            )

    def __init__(self, ninputs, noutputs,
                 hlayers_sizes, learning_rate, momentum, beta=0.9):

        self.layers_sizes = [ninputs + 1] + [s + 1 for s in hlayers_sizes] + [noutputs]
        self.nlayers = len(self.layers_sizes)
        self.learning_rate = learning_rate
        self.beta = beta
        self.momentum = momentum

        self.layers = \
            [np.zeros((1, s), dtype=float) for s in self.layers_sizes]

        self.weights = []
        self.__init_weights()

        self.delta_weights = []
        self.__init_delta_weights()

    def activate(self, input):

        self.layers[0] = np.array([input + [1]], float)

        for i in range(self.nlayers - 1):
            self.layers[i][0][-1] = 1

        for i in range(1, self.nlayers):
            self.layers[i] = self.__sigmoid(
                np.dot(self.layers[i - 1], self.weights[i - 1]))

        return self.layers[-1][0]

    def correction(self, output):

        error = (np.array([output], float) - self.layers[-1])
        abs_error = sum(error[0] ** 2)

        for i in range(self.nlayers - 1, 0, -1):
            error *= self.beta * self.layers[i] * (1 - self.layers[i])
            self.delta_weights[i - 1] += \
                self.learning_rate * np.dot(self.layers[i - 1].transpose(), error)

            error = np.dot(error, self.weights[i - 1].transpose())

        return abs_error

    def error(self, input, output):

        self.activate(input)
        error = (np.array([output], float) - self.layers[-1])
        abs_error = sum(error[0] ** 2)

        return abs_error

    def adaptation(self):

        for i in range(0, len(self.weights)):
            self.weights[i] += self.delta_weights[i]
            self.delta_weights[i] *= self.momentum


class Experiment(object):

    def __init__(self, training_set, validator, min_error, max_loops, file_handler, runs):

        self.validator = validator
        self.runs = runs
        self.file_handler = file_handler
        self.min_error = min_error
        self.max_loops = max_loops
        self.training_set = training_set

    def run(self, hlayers_sizes, learning_rate, momentum, run_callback=None):

        for run in range(1, self.runs + 1):
            mlp = MultiLayerPerceptron(
                len(self.training_set[0][0]),
                len(self.training_set[0][1]),
                hlayers_sizes, learning_rate, momentum
            )
            (ok, error, loops, partial_errors) = self.validator.validate(mlp, self.training_set, self.min_error, self.max_loops)

            if run_callback:
                run_callback(run, ok, error, loops, partial_errors)

            self.file_handler.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(hlayers_sizes, learning_rate, momentum, ok, error, loops))
