import sys

import numpy as np
import pandas as pd

inp_train = pd.read_csv(sys.argv[1], header=None)
out_train = pd.read_csv(sys.argv[2], header=None)
inp_test = pd.read_csv(sys.argv[3], header=None)
inp_x_train = np.array(inp_train, dtype=float)
out_y_train = np.array(out_train)
y_train_enc = np.eye(10)[out_y_train.reshape(-1)]

inp_x_test = np.array(inp_test, dtype=float)

inp_x_train /= 255.0
inp_x_test /= 255.0


class NeuralNetwork:
    def __init__(self, network, learning_rate, epochs, batch_size):
        self.network = network
        self.w1 = np.random.randn(network[1], network[0]) / np.sqrt(network[0])
        self.b1 = np.zeros((network[1], 1)) / np.sqrt(network[0])
        self.w2 = np.random.randn(network[2], network[1]) / np.sqrt(network[1])
        self.b2 = np.zeros((network[2], 1)) / np.sqrt(network[1])
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.store = {}

    def forward_propagation(self, x):
        # store forward computed values for backpropagation
        self.store["input"] = x
        self.store["sum1"] = np.matmul(self.w1, self.store["input"].T) + self.b1
        self.store["activation1"] = sigmoid(self.store["sum1"])
        self.store["sum2"] = np.matmul(self.w2, self.store["activation1"]) + self.b2
        self.store["activation2"] = softmax(self.store["sum2"])
        return self.store["activation2"]

    def backward_propagation(self, actual, predicted):
        derivative_l = predicted - actual.T
        derivative_w2 = np.matmul(derivative_l, self.store["activation1"].T) / (actual.shape[0])
        derivative_b2 = np.sum(derivative_l, axis=1, keepdims=True) / (actual.shape[0])
        derivative_activation1 = np.matmul(self.w2.T, derivative_l)
        derivative_sum1 = derivative_activation1 * sigmoid_derivative(self.store["sum1"])
        derivative_w1 = np.matmul(derivative_sum1, self.store["input"]) / (actual.shape[0])
        derivative_b1 = np.sum(derivative_sum1, axis=1, keepdims=True) / (actual.shape[0])
        # stochastic gradient descent
        self.w1 = self.w1 - self.learning_rate * derivative_w1
        self.w2 = self.w2 - self.learning_rate * derivative_w2
        self.b1 = self.b1 - self.learning_rate * derivative_b1
        self.b2 = self.b2 - self.learning_rate * derivative_b2

    def train(self, x_train, y_train, x_test):
        batches = x_train.shape[0] // self.batch_size

        for _ in range(self.epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_perm = x_train[permutation]
            y_perm = y_train[permutation]

            for j in range(batches):
                x = x_perm[j * self.batch_size: min(j * self.batch_size + self.batch_size, x_train.shape[0] - 1)]
                y = y_perm[j * self.batch_size: min(j * self.batch_size + self.batch_size, x_train.shape[0] - 1)]
                forward = self.forward_propagation(x)
                self.backward_propagation(y, forward)

        test_predicted = self.forward_propagation(x_test)
        predictions = test_predicted.T.argmax(axis=-1)
        df = pd.DataFrame(predictions)
        df.to_csv('test_predictions.csv', index=False, header=False)


def sigmoid(inp):
    return 1 / (1 + np.exp(-inp))


def sigmoid_derivative(inp):
    return np.exp(-inp) / (np.exp(-inp) + 1) ** 2


def categorical_cross_entropy(actual, predicted):
    return -(1. / actual.shape[0]) * np.sum(np.multiply(actual.T, np.log(predicted)))


def softmax(inp):
    return np.exp(inp - inp.max()) / np.sum(np.exp(inp - inp.max()), axis=0)


# neural network with 1 hidden layer
nn = NeuralNetwork([784, 100, 10], 0.1, 50, 50)
nn.train(inp_x_train, y_train_enc, inp_x_test)
