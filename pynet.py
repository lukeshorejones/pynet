import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


# derivative of sigmoid(x) is technically d_sigmoid(sigmoid(x))
def d_sigmoid(y):
    return y * (1 - y)


class NeuralNetwork:
    def __init__(self, input_count, hidden_count, output_count, learn_rate):
        self.input_count = input_count
        self.hidden_count = hidden_count
        self.output_count = output_count
        self.learn_rate = learn_rate

        self.inputs = None
        self.hidden = None
        self.outputs = None

        # initialise weights as matrices with random float values -1 to 1
        # biases are like nodes with value 1 and their own weights
        self.weights_1 = np.random.rand(self.hidden_count,
                                        self.input_count) * 2 - 1

        self.biases_1 = np.random.rand(self.hidden_count, 1) * 2 - 1

        self.weights_2 = np.random.rand(self.output_count,
                                        self.hidden_count) * 2 - 1

        self.biases_2 = np.random.rand(self.output_count, 1) * 2 - 1

    # produce outputs from given inputs
    def assess(self, inputs):
        self.inputs = np.array([inputs]).T
        self.hidden = sigmoid(self.weights_1 @ self.inputs + self.biases_1)
        self.outputs = sigmoid(self.weights_2 @ self.hidden + self.biases_2)

        return self.outputs.flatten().tolist()

    # assess, then improve weights given target outputs
    def train(self, inputs, targets):
        outputs_list = self.assess(inputs)

        output_errors = targets - self.outputs
        w2_gradients = d_sigmoid(self.outputs) * output_errors * self.learn_rate
        w2_deltas = w2_gradients @ self.hidden.T
        self.weights_2 += w2_deltas
        self.biases_2 += w2_gradients

        hidden_errors = self.weights_2.T @ output_errors
        w1_gradients = d_sigmoid(self.hidden) * hidden_errors * self.learn_rate
        w1_deltas = w1_gradients @ self.inputs.T
        self.weights_1 += w1_deltas
        self.biases_1 += w1_gradients

        return outputs_list
