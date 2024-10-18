import numpy as np
from abc import ABC, abstractmethod


# Superclass for activation functions
class ActivationFunction(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def forward_pass(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class tanH(ActivationFunction):
    def __init__(self):
        super().__init__("tanh")

    def forward_pass(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2



class ReLU(ActivationFunction):

    def __init__(self):
        super().__init__("relu")

    def forward_pass(self, x):
        return np.maximum(0, x)

    def derivative(self, x):

        return np.where(self.forward_pass(x) >= 0, 1, 0)


# Step subclass
class Step(ActivationFunction):

    def __init__(self):
        super().__init__("step")

    def forward_pass(self, x):
        return np.where(x >= 0, 1, 0)

    def derivative(self, x):
        return np.zeros_like(x)  # Derivative is 0 everywhere


class Sigmoid(ActivationFunction):

    def __init__(self):
        super().__init__("sigmoid")

    def forward_pass(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sig = self.forward_pass(x)

        return 1 - sig

# lloss functions
class MeanSquaredError:

    def calculate(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

# perceptron class
class Perceptron:
    def __init__(
        self,
        units,
        activation_function,
        input_shape=None,
        weights=None,
        bias=None,
        name=None,
    ):

        self.activation_function = activation_function
        self.units = units
        self.name = name
        self.input_shape = input_shape
        self.weights = weights
        # for loading weights in when saved
        if weights is not None and bias is not None and input_shape is not None:
            self.input_shape = input_shape
            self.weights = weights
            self.bias = bias

    def build(self, input_shape):

        if self.input_shape is None:
            self.input_shape = input_shape

        if self.weights is None:
            self.weights = np.random.uniform(
                low=-1, high=1, size=(self.input_shape, self.units)
            )

            self.bias = [[1] * self.units]  # np.random.rand(1,self.units)

    def forward_pass(self, input_vector):

        return self.activation_function.forward_pass(
            np.dot(input_vector, self.weights) + self.bias
        )

    def backward_pass(self, delta, input_vector, lr):
        self.weights -= lr * np.dot(input_vector.T, delta)
        self.bias -= lr * np.sum(delta, axis=0, keepdims=True)

    def calculate_derivative(self, x):
        return self.activation_function.derivative(x)

    def summary(self):
        print(f"units: {self.units}, input_shape = {self.input_shape,self.units}, activation = {self.activation_function.name}")

    def get_weights(self):
        return self.weights


class MLP:
    def __init__(self, input_shape, name=None):

        self.name = name
        self.input_shape = input_shape
        self.layers = []

    def add_layer(self, perceptron):

        if len(self.layers) == 0:
            perceptron.build(1)  # self.input_shape? why
        else:
            perceptron.build(self.layers[-1].units)
        self.layers.append(perceptron)

    def compile(self, loss_function):
        self.loss_function = loss_function

    def forward_pass(self, input):

        hidden_outputs = []
        output = None

        for layer in self.layers:

            output = layer.forward_pass(input)
            hidden_outputs.append(output)
            input = output

        return hidden_outputs, output

    def backward_pass(self, input, output, target, hidden_outputs, learning_rate=0.01):
        out_layer = self.layers[-1]
        out_deltas = self.loss_function.derivative(target, output) * out_layer.activation_function.derivative(output)

        out_layer.backward_pass(out_deltas, hidden_outputs[-2], learning_rate)

        for layer_ind in reversed(range(len(self.layers) - 1)):
            hidden_layer = self.layers[layer_ind]
            next_layer = self.layers[layer_ind + 1]

            in_deltas = np.dot(out_deltas, next_layer.weights.T) * hidden_layer.activation_function.derivative(hidden_outputs[layer_ind])

            if layer_ind > 0:
                hidden_layer.backward_pass(
                    in_deltas, hidden_outputs[layer_ind - 1], learning_rate
                )
            else:
                hidden_layer.backward_pass(in_deltas, input, learning_rate)

            out_deltas = in_deltas

    def fit(self, epochs, x_train, y_train, learning_rate=0.01):
        for i in range(epochs):
            epoch_loss = 0
            for input_vector, target in zip(x_train, y_train):

                hidden_outs, output = self.forward_pass(input_vector)
                loss = self.loss_function.calculate(target, output)
                epoch_loss += loss

                self.backward_pass(
                    input_vector.reshape(1, -1),
                    output,
                    target.reshape(1, -1),
                    hidden_outs,
                    learning_rate,
                )

            print(f"Epoch {i + 1}/{epochs}, Loss: {epoch_loss / len(x_train)}")

    def accuracy(self, x, y):

        predictions = []

        for input_vector in training_inputs:
            predicted_output = mlp1.forward_pass(input_vector)[1]

            predicted_value = 1 if predicted_output[0][0] >= 0.5 else 0
            predictions.append(predicted_value)

        # accuracy
        correct_predictions = sum(
            1
            for pred, true in zip(predictions, training_targets.flatten())
            if pred == true
        )
        accuracy = correct_predictions / len(training_targets) * 100
        return accuracy

    def summary(self):
        for layer in self.layers:
            layer.summary()


if __name__ == "__main__":

    # XOR problem
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_targets = np.array([[0], [1], [1], [0]])

    mlp1 = MLP(1, "test")

    mlp1.add_layer(Perceptron(2, Sigmoid(), input_shape=2, name="layer 1"))
    mlp1.add_layer(Perceptron(4, tanH(), name="layer 2"))

    mlp1.add_layer(Perceptron(1, Sigmoid(), name="layer 3"))
    mlp1.compile(MeanSquaredError())

    # print(mlp1.forward_pass(input)[1])
    before = mlp1.accuracy(training_inputs, training_targets)
    mlp1.fit(40_000, training_inputs, training_targets)

    errors = []

    for input, target in zip(training_inputs, training_targets):
        predicted_output = mlp1.forward_pass(input)[1]

        predicted_value = predicted_output[0][0]
        error = predicted_value - target

        errors.append(error**2)
        print(predicted_value)

    mse = np.sqrt(np.mean(errors))
    print(f"Mean Squared Error: {mse}")

    print(before)
    print(mlp1.accuracy(training_inputs, training_targets))
    print("-------------")
    mlp1.summary()
